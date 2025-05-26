from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union
import re # Added for parsing EP labels

import networkx as nx
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import torch # Added for GEDGW conversion
import dgl # Added for GEDGW test

# Conditional import for pydot
try:
    import pydot
    from networkx.drawing.nx_pydot import graphviz_layout
    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False

import numpy as np
if not hasattr(np, 'alltrue'): # For older numpy versions
    np.alltrue = np.all

# --- Imports from other files in the 'src' directory ---
from models import GEDGW
# Ensure 'ot' (Python Optimal Transport library) is installed for models.GEDGW

class NodeRole(Enum):
    COMPARTMENT = auto()
    EXTERIOR = auto()
    EXPR_OP = auto()
    EXPR_FUNCTION = auto()
    EXPR_SYMBOL = auto()
    EXPR_CONSTANT = auto()
    EDGE_PROXY = auto()

    def prefix(self) -> str:
        return {
            NodeRole.COMPARTMENT: "C",
            NodeRole.EXTERIOR: "X",
            NodeRole.EXPR_OP: "O",
            NodeRole.EXPR_FUNCTION: "F",
            NodeRole.EXPR_SYMBOL: "E",
            NodeRole.EXPR_CONSTANT: "K",
            NodeRole.EDGE_PROXY: "EP",
        }[self]

class InteractionType(Enum):
    FLOW = "flow"

@dataclass
class NodeData:
    role: NodeRole
    label: str
    raw_label: str
    sympy_obj: Optional[sp.Basic] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = dict(role=self.role, label=self.label, raw_label=self.raw_label, **self.attrs)
        if self.sympy_obj is not None:
            d["sympy_str"] = str(self.sympy_obj)
        return d

@dataclass
class Interaction:
    source: str
    target: str
    int_type: InteractionType
    expr_root: str

StateVarSet = Set[str]
ParamSet = Set[str]


def _depends_on(expr: sp.Basic, state_vars: StateVarSet) -> bool:
    if not isinstance(expr, sp.Basic):
        return False
    return any(str(s) in state_vars for s in expr.free_symbols)

def _expand_state_dependent_products(expr: sp.Basic, state_vars: StateVarSet) -> sp.Basic:
    if not isinstance(expr, sp.Basic): return expr
    processed_args = [_expand_state_dependent_products(a, state_vars) for a in expr.args]
    try: expr = expr.func(*processed_args)
    except Exception:
        try: expr = expr.func(*expr.args)
        except Exception: return expr
    if not isinstance(expr, sp.Mul): return expr
    non_sums, sums_to_expand = [], []
    for arg in expr.args:
        if isinstance(arg, sp.Add) and _depends_on(arg, state_vars): sums_to_expand.append(arg)
        else: non_sums.append(arg)
    if not sums_to_expand: return expr
    prod_root = sp.Mul(*non_sums) if non_sums else sp.Integer(1)
    current = prod_root
    for s in sums_to_expand:
        current = sp.expand(current * s)
        current = _expand_state_dependent_products(current, state_vars)
    return current

def _flatten_add(expr: sp.Basic) -> sp.Basic:
    if not isinstance(expr, sp.Basic) or not expr.args: return expr
    processed_args = [_flatten_add(a) for a in expr.args]
    if not isinstance(expr, sp.Add):
        try:
            if any(pa is not oa for pa, oa in zip(processed_args, expr.args)): return expr.func(*processed_args)
            else: return expr
        except Exception: return expr
    flat = []
    made_change_at_this_level = False
    for a in processed_args:
        if isinstance(a, sp.Add):
            flat.extend(a.args); made_change_at_this_level = True
        else: flat.append(a)
    children_changed = any(pa is not oa for pa, oa in zip(processed_args, expr.args))
    if made_change_at_this_level or children_changed: return sp.Add(*flat, evaluate=False)
    else: return expr

def canonicalize_expression(expr: sp.Basic,
                            state_vars: StateVarSet,
                            params: ParamSet,
                            level: str = "basic") -> sp.Basic:
    if not isinstance(expr, sp.Basic): return expr
    try: e = _expand_state_dependent_products(expr, state_vars)
    except Exception as exp_e: print(f"Warning: _expand_state_dependent_products failed for {expr}: {exp_e}. Using original."); e = expr
    try: e = _flatten_add(e)
    except Exception as flat_e: print(f"Warning: _flatten_add failed for {e}: {flat_e}. Using result from expansion.");
    original_for_simplify = e
    if level == "basic":
        try:
            e_cancel = sp.cancel(e)
            if e_cancel != original_for_simplify: e = e_cancel
            else: e = sp.simplify(e)
        except Exception as e_simp: print(f"Warning: Basic simplification (cancel/simplify) failed for {original_for_simplify}: {e_simp}. Using pre-simplification form."); e = original_for_simplify
    elif level == "full":
        try: e = sp.simplify(e)
        except Exception as e_full: print(f"Warning: Full simplification failed for {original_for_simplify}: {e_full}. Using pre-simplification form."); e = original_for_simplify
    elif level == "ratsimp":
         try: e = sp.ratsimp(e)
         except Exception as e_rat: print(f"Warning: Rational simplification (ratsimp) failed for {original_for_simplify}: {e_rat}. Using pre-simplification form."); e = original_for_simplify
    elif level == "none": pass
    else: raise ValueError(f"Unknown simplification level: {level}")
    return e


class EpiModelGraph:
    EXTERIOR_NODE_LABEL = "EXTERIOR"

    def __init__(self, name: str, state_vars: Iterable[str], params: Iterable[str]):
        self.name = name
        _state_vars = set(state_vars)
        if self.EXTERIOR_NODE_LABEL in _state_vars:
            raise ValueError(f"State variable name '{self.EXTERIOR_NODE_LABEL}' is reserved.")
        self.state_vars: StateVarSet = _state_vars
        self.params: ParamSet = set(params) # Store param names as strings

        # Store canonical Sympy Symbol objects for declared parameters
        self.param_symbol_objects: Dict[str, sp.Symbol] = {p_name: sp.Symbol(p_name) for p_name in self.params}

        self.g: nx.DiGraph = nx.DiGraph(name=name)
        self._ids = itertools.count()
        self._node_cache: Dict[Tuple[NodeRole, str], str] = {}
        self._shared_param_symbol_node_cache: Dict[sp.Symbol, str] = {} # Key is canonical sp.Symbol object
        self._shared_constant_node_cache: Dict[sp.Basic, str] = {}

        self.exterior_node_id = self._add_node(
            role=NodeRole.EXTERIOR,
            raw_label=self.EXTERIOR_NODE_LABEL,
            label=f"{NodeRole.EXTERIOR.prefix()}:{self.EXTERIOR_NODE_LABEL}",
            always_create_new=False
        )

        for comp in self.state_vars:
            self._add_node(role=NodeRole.COMPARTMENT,
                           raw_label=comp,
                           label=f"{NodeRole.COMPARTMENT.prefix()}:{comp}",
                           always_create_new=False)

    def _new_id(self, role: NodeRole, raw: str) -> str:
        raw_simple = "".join(c for c in raw if c.isalnum())[:10]
        return f"{role.prefix()}_{raw_simple}_{next(self._ids)}"

    def _add_node(self, *,
                  role: NodeRole,
                  raw_label: str,
                  label: Optional[str] = None,
                  sympy_obj: Optional[sp.Basic] = None,
                  always_create_new: bool = False,
                  **attrs) -> str:
        cache_key = (role, raw_label)
        if not always_create_new and role in (NodeRole.COMPARTMENT, NodeRole.EXTERIOR):
            if cache_key in self._node_cache:
                return self._node_cache[cache_key]

        display_label = label if label is not None else f"{role.prefix()}:{raw_label}"
        nid = self._new_id(role, raw_label)
        # Ensure 'attrs' from NodeData is used if passed, otherwise use the kwargs 'attrs'
        node_attrs = attrs # attrs from the _add_node call (e.g. symbol_type)
        data = NodeData(role=role, label=display_label, raw_label=raw_label, sympy_obj=sympy_obj, attrs=node_attrs)
        self.g.add_node(nid, **data.as_dict())


        if not always_create_new and role in (NodeRole.COMPARTMENT, NodeRole.EXTERIOR):
             self._node_cache[cache_key] = nid
        return nid

    def add_compartment(self, name: str, **attrs) -> str:
        if name == self.EXTERIOR_NODE_LABEL: raise ValueError(f"Cannot add compartment with reserved name '{self.EXTERIOR_NODE_LABEL}'.")
        if name in self.state_vars:
             print(f"Warning: Compartment '{name}' already exists.")
             return self._get_node_id(NodeRole.COMPARTMENT, name)
        self.state_vars.add(name)
        nid = self._add_node(role=NodeRole.COMPARTMENT, raw_label=name, label=f"C:{name}", always_create_new=False, **attrs)
        return nid

    def _get_node_id(self, role: NodeRole, raw_label: str) -> str:
        cache_key = (role, raw_label)
        if role not in (NodeRole.COMPARTMENT, NodeRole.EXTERIOR) or cache_key not in self._node_cache:
            # This function is primarily for COMPARTMENT/EXTERIOR.
            # Parameters are handled by _shared_param_symbol_node_cache using canonical symbol objects.
            raise KeyError(f"Node with role '{role}' and raw label '{raw_label}' not found in standard cache. Was it added?")
        return self._node_cache[cache_key]

    def _lookup_source_target_node(self, comp_or_exterior: str) -> str:
        if comp_or_exterior == self.EXTERIOR_NODE_LABEL: return self.exterior_node_id
        elif comp_or_exterior in self.state_vars: return self._get_node_id(NodeRole.COMPARTMENT, comp_or_exterior)
        else: raise ValueError(f"Unknown source/target '{comp_or_exterior}'. Must be a registered compartment or '{self.EXTERIOR_NODE_LABEL}'.")

    def add_interaction(self,
                        source: str,
                        target: str,
                        expr: sp.Expr,
                        canonical_level: str = "none") -> Interaction:
        int_type = InteractionType.FLOW
        src_nid = self._lookup_source_target_node(source)
        tgt_nid = self._lookup_source_target_node(target)

        if src_nid == tgt_nid:
             if source == self.EXTERIOR_NODE_LABEL: raise ValueError("Cannot add interaction from EXTERIOR to itself.")
             else: print(f"Warning: Adding interaction from compartment '{source}' to itself.")

        canon_expr = canonicalize_expression(expr, self.state_vars, self.params, level=canonical_level)
        root_nid = self._expr_to_subgraph(canon_expr)

        self.g.add_edge(src_nid, root_nid, kind=int_type.value, type="interaction_start")
        self.g.add_edge(root_nid, tgt_nid, kind=int_type.value, type="interaction_end")

        return Interaction(source=source, target=target, int_type=int_type, expr_root=root_nid)

    def _add_or_update_edge(self, u: str, v: str, **attrs_to_set: Any) -> None:
        if not self.g.has_edge(u, v):
            self.g.add_edge(u, v, **attrs_to_set)
        else:
            edge_data = self.g.edges[u, v]
            for key, new_value in attrs_to_set.items():
                if key not in edge_data:
                    edge_data[key] = new_value
                else:
                    existing_value = edge_data[key]
                    if existing_value == new_value:
                        continue

                    if key == 'order':
                        current_orders = []
                        if isinstance(existing_value, list): current_orders.extend(existing_value)
                        else: current_orders.append(existing_value)
                        
                        new_orders_to_add = []
                        if isinstance(new_value, list): new_orders_to_add.extend(new_value)
                        else: new_orders_to_add.append(new_value)

                        changed = False
                        for no_val in new_orders_to_add:
                            if no_val not in current_orders:
                                current_orders.append(no_val)
                                changed = True
                        if changed:
                            edge_data[key] = sorted(list(set(current_orders)))
                    elif key == 'kind':
                        current_kinds = []
                        if isinstance(existing_value, list): current_kinds.extend(existing_value)
                        else: current_kinds.append(existing_value)
                        
                        new_kinds_to_add = []
                        if isinstance(new_value, list): new_kinds_to_add.extend(new_value)
                        else: new_kinds_to_add.append(new_value)

                        changed = False
                        for nk_val in new_kinds_to_add:
                            if nk_val not in current_kinds:
                                current_kinds.append(nk_val)
                                changed = True
                        
                        if changed:
                            updated_kinds = sorted(list(set(current_kinds)))
                            edge_data[key] = updated_kinds
                            if len(updated_kinds) > 1 :
                                print(f"Warning: Edge ({u}, {v}) attribute 'kind' merged to list: {updated_kinds}. Original: {existing_value}, Added: {new_value}")
                        elif len(current_kinds) == 1 and current_kinds[0] != existing_value and not isinstance(existing_value, list) :
                             edge_data[key] = current_kinds[0]
                        elif len(current_kinds) == 1 and isinstance(existing_value, list) and len(existing_value) > 1:
                             edge_data[key] = current_kinds
                    else:
                        if edge_data.get(key) != new_value:
                            edge_data[key] = new_value

    def _expr_to_subgraph(self, expr: sp.Basic) -> str:
        local_expr_node_cache: Dict[sp.Basic, str] = {}

        def _rec(e: sp.Basic) -> Tuple[str, bool]:
            if not isinstance(e, sp.Basic):
                 try: hash(e); cache_key = e
                 except TypeError: cache_key = repr(e)
                 if cache_key in local_expr_node_cache: return local_expr_node_cache[cache_key], False
                 raw_label_nonsympy = str(e)
                 print(f"Warning: Non-Sympy object '{raw_label_nonsympy}' encountered. Treating as Constant.")
                 nid_nonsympy = self._add_node(role=NodeRole.EXPR_CONSTANT,
                                      raw_label=raw_label_nonsympy,
                                      label=f"K:NonSympy_{type(e).__name__}",
                                      sympy_obj=None,
                                      always_create_new=True)
                 local_expr_node_cache[cache_key] = nid_nonsympy
                 return nid_nonsympy, False

            if isinstance(e, sp.Symbol):
                raw_label_str_e = str(e)
                if raw_label_str_e in self.state_vars:
                    comp_nid = self._get_node_id(NodeRole.COMPARTMENT, raw_label_str_e)
                    return comp_nid, True

            # Check shared caches using canonical objects for params/constants
            canonical_param_obj_for_cache = None
            is_declared_param = False
            if isinstance(e, sp.Symbol):
                raw_label_str_e = str(e)
                if raw_label_str_e in self.params: # Is its string name a declared parameter?
                    is_declared_param = True
                    canonical_param_obj_for_cache = self.param_symbol_objects[raw_label_str_e]
                    if canonical_param_obj_for_cache in self._shared_param_symbol_node_cache:
                        shared_nid = self._shared_param_symbol_node_cache[canonical_param_obj_for_cache]
                        local_expr_node_cache[e] = shared_nid # Link current 'e' to shared node locally
                        return shared_nid, False

            if isinstance(e, (sp.Number, sp.NumberSymbol)):
                # For constants, 'e' itself is usually canonical if it's the same value/type
                if e in self._shared_constant_node_cache:
                    shared_nid = self._shared_constant_node_cache[e]
                    local_expr_node_cache[e] = shared_nid
                    return shared_nid, False

            if e in local_expr_node_cache:
                return local_expr_node_cache[e], False

            nid: str
            attrs: Dict[str, Any] = {}

            if isinstance(e, sp.Integral):
                role_integral = NodeRole.EXPR_FUNCTION
                raw_label_integral = e.func.__name__
                node_label_integral = f"{role_integral.prefix()}:{raw_label_integral}"
                nid = self._add_node(role=role_integral, raw_label=raw_label_integral, label=node_label_integral,
                                     sympy_obj=e, always_create_new=True, **attrs)
                local_expr_node_cache[e] = nid
                if e.args:
                    integrand_expr = e.args[0]
                    integrand_nid, _ = _rec(integrand_expr)
                    self._add_or_update_edge(nid, integrand_nid, kind="integrand", order=0)
                for dim_idx, limit_spec in enumerate(e.limits):
                    limit_spec_arg_order = dim_idx + 1
                    is_python_tuple = isinstance(limit_spec, tuple)
                    is_sympy_tuple = hasattr(sp, 'Tuple') and isinstance(limit_spec, sp.Tuple)
                    if not (is_python_tuple or is_sympy_tuple) or len(limit_spec) == 0:
                        print(f"Warning: Integral {nid} (Expr: {e}) has an invalid limit specification: {limit_spec}. Skipping.")
                        continue
                    var_symbol = limit_spec[0]
                    var_nid, _ = _rec(var_symbol)
                    self._add_or_update_edge(nid, var_nid, kind="integration_variable", order=limit_spec_arg_order, dim_index=dim_idx)
                    if len(limit_spec) > 1 and limit_spec[1] is not None:
                        lower_limit_expr = limit_spec[1]
                        lower_nid, _ = _rec(lower_limit_expr)
                        self._add_or_update_edge(nid, lower_nid, kind="lower_limit", order=limit_spec_arg_order, dim_index=dim_idx)
                    if len(limit_spec) > 2 and limit_spec[2] is not None:
                        upper_limit_expr = limit_spec[2]
                        upper_nid, _ = _rec(upper_limit_expr)
                        self._add_or_update_edge(nid, upper_nid, kind="upper_limit", order=limit_spec_arg_order, dim_index=dim_idx)
                return nid, False

            force_creation = True
            role_general: NodeRole
            raw_label_general: str = str(e)
            node_label_general: str

            if e.args:
                raw_label_general = e.func.__name__
                if issubclass(e.func, sp.Function):
                    role_general = NodeRole.EXPR_FUNCTION
                    node_label_general = f"F:{raw_label_general}"
                else:
                    role_general = NodeRole.EXPR_OP
                    node_label_general = f"O:{raw_label_general}"
            else: # Leaf node
                current_symbol_str = str(e) # Use str(e) for label generation
                if isinstance(e, sp.Symbol):
                    role_general = NodeRole.EXPR_SYMBOL
                    if is_declared_param: # Correctly identified as a parameter string name
                        force_creation = False # Will use shared cache
                        attrs['symbol_type'] = 'param'
                        node_label_general = f"P:{current_symbol_str}"
                    else: # Not a declared state var, not a declared param
                        attrs['symbol_type'] = 'unknown'
                        node_label_general = f"U:{current_symbol_str}"
                elif isinstance(e, (sp.Number, sp.NumberSymbol)):
                     role_general = NodeRole.EXPR_CONSTANT
                     node_label_general = f"K:{current_symbol_str}"
                     force_creation = False # Will use shared cache
                else:
                     role_general = NodeRole.EXPR_SYMBOL
                     print(f"Warning: Unhandled leaf expression type: {type(e)}. Treating as Symbol.")
                     node_label_general = f"U:{current_symbol_str}"
                     attrs['symbol_type'] = 'unhandled'
            
            nid = self._add_node(role=role_general, raw_label=raw_label_general, label=node_label_general,
                                 sympy_obj=e, always_create_new=force_creation, **attrs)
            
            # Add to shared caches IF it's a parameter/constant AND it's the first time (force_creation=False implies this)
            if is_declared_param and not force_creation and canonical_param_obj_for_cache:
                self._shared_param_symbol_node_cache[canonical_param_obj_for_cache] = nid
            elif role_general == NodeRole.EXPR_CONSTANT and not force_creation:
                self._shared_constant_node_cache[e] = nid # Use 'e' itself as key for constants
            
            local_expr_node_cache[e] = nid

            if e.args:
                is_pow = isinstance(e, sp.Pow)
                for i, arg in enumerate(e.args):
                    child_id, child_is_compartment = _rec(arg)
                    edge_kind = "operand"
                    if is_pow:
                        if i == 0: edge_kind = "base"
                        elif i == 1: edge_kind = "exponent"
                        else: print(f"Warning: Pow op {nid} unexpected arg index {i}.")
                    elif child_is_compartment:
                         edge_kind = "state_dependency"
                    self._add_or_update_edge(nid, child_id, kind=edge_kind, order=i)
            
            return nid, False

        root_nid, _ = _rec(expr)
        return root_nid

    def to_networkx(self) -> nx.DiGraph:
        return self.g

    def to_undirected_levi_graph(self) -> nx.Graph:
        undirected_g = nx.Graph(name=f"undirected_levi_{self.name}")
        for node_id, node_data in self.g.nodes(data=True):
            undirected_g.add_node(node_id, **node_data)
        edge_proxy_counter = 0
        for u, v, original_edge_data in self.g.edges(data=True):
            ep_node_id = self._new_id(NodeRole.EDGE_PROXY, f"edge_{u}_to_{v}")
            edge_proxy_counter +=1
            ep_label_parts = []
            if 'kind' in original_edge_data:
                kind_val = original_edge_data['kind']
                if isinstance(kind_val, list): ep_label_parts.append(f"kind:{','.join(map(str,kind_val))}")
                else: ep_label_parts.append(f"kind:{kind_val}")
            if 'type' in original_edge_data: ep_label_parts.append(f"type:{original_edge_data['type']}")
            if 'order' in original_edge_data:
                order_val = original_edge_data['order']
                if isinstance(order_val, list): ep_label_parts.append(f"order:{','.join(map(str,order_val))}")
                else: ep_label_parts.append(f"order:{order_val}")
            if 'dim_index' in original_edge_data:
                ep_label_parts.append(f"dim:{original_edge_data['dim_index']}")
            ep_display_label = f"{NodeRole.EDGE_PROXY.prefix()}:" + (";".join(ep_label_parts) if ep_label_parts else "edge")
            ep_node_attrs = {
                'role': NodeRole.EDGE_PROXY,
                'label': ep_display_label,
                'raw_label': f"proxy_for_edge_from_{u}_to_{v}",
                'original_attributes': dict(original_edge_data),
                'original_source_id': u,
                'original_target_id': v
            }
            undirected_g.add_node(ep_node_id, **ep_node_attrs)
            undirected_g.add_edge(u, ep_node_id, link_type="source_to_proxy")
            undirected_g.add_edge(ep_node_id, v, link_type="proxy_to_target")
        return undirected_g

    def summary(self) -> str:
        comps = sorted([d["raw_label"] for n, d in self.g.nodes(data=True) if d["role"] == NodeRole.COMPARTMENT])
        interactions = []
        num_expr_nodes = 0
        num_comp_nodes = 0
        num_ext_nodes = 0
        num_param_nodes = 0

        for n, d in self.g.nodes(data=True):
            role = d.get("role")
            node_attrs = d.get("attrs", {}) # Get the 'attrs' dict stored in NodeData
            if role == NodeRole.COMPARTMENT: num_comp_nodes += 1
            elif role == NodeRole.EXTERIOR: num_ext_nodes += 1
            elif role == NodeRole.EXPR_SYMBOL and node_attrs.get('symbol_type') == 'param': # Check 'symbol_type' within 'attrs'
                num_param_nodes +=1
            elif role in (NodeRole.EXPR_OP, NodeRole.EXPR_FUNCTION, NodeRole.EXPR_SYMBOL, NodeRole.EXPR_CONSTANT): num_expr_nodes +=1

        for u, v, d_edge in self.g.edges(data=True):
            if d_edge.get("type") == "interaction_start":
                 try:
                     u_data = self.g.nodes[u]
                     expr_root_nid = v
                     end_edge_target = None
                     for _, target_node, edge_data_out in self.g.out_edges(expr_root_nid, data=True):
                         if edge_data_out.get("type") == "interaction_end":
                             end_edge_target = target_node
                             break
                     if end_edge_target:
                         tgt_data = self.g.nodes[end_edge_target]
                         start_kind = d_edge.get("kind", "unknown_flow")
                         if u_data["role"] in (NodeRole.COMPARTMENT, NodeRole.EXTERIOR) and \
                            tgt_data["role"] in (NodeRole.COMPARTMENT, NodeRole.EXTERIOR):
                             interactions.append((u_data["raw_label"], tgt_data["raw_label"], start_kind))
                     else: print(f"Warning: Could not find 'interaction_end' edge for expr root {expr_root_nid} starting from {u_data['raw_label']}.")
                 except KeyError as e: print(f"Warning: Node data missing during summary for edge ({u}, {v}): {e}")
                 except Exception as e: print(f"Warning: Error processing edge ({u}, {v}) for summary: {e}")
        
        return (f"EpiModelGraph(name={self.name!r}, "
                f"compartments={comps} ({num_comp_nodes}), "
                f"params_nodes={num_param_nodes}, "
                f"exterior_nodes={num_ext_nodes}, "
                f"interactions={len(interactions)}, "
                f"other_expression_nodes={num_expr_nodes})\\n"
                f"Note: Other expression nodes include Operations (O:), Functions (F:), Unknowns (U:), and Constants (K:). Pow uses 'base'/'exponent' edges. State variable dependencies use 'state_dependency' edges. Integrals use 'integrand', 'integration_variable', 'lower_limit', 'upper_limit' edges.")

    def get_conceptual_label_for_node(self, node_id: str, node_data: Dict[str, Any]) -> str:
        role = node_data.get('role')
        original_display_label = node_data.get('label', f"NO_LABEL_FOR_{node_id}")

        if not isinstance(role, NodeRole):
            try: role = NodeRole[str(role).split('.')[-1].upper()]
            except: role = None

        if role == NodeRole.EDGE_PROXY:
            if not original_display_label.startswith("EP:"):
                print(f"Warning: EP node {node_id} has unexpected label format: {original_display_label}. Using as is.")
                return original_display_label
            ep_content_str = original_display_label[len("EP:"):]
            attributes = ep_content_str.split(';')
            current_kinds_in_ep = []
            for attr_part in attributes:
                if attr_part.startswith("kind:"):
                    kind_val_str = attr_part.split(":", 1)[1]
                    current_kinds_in_ep.extend(k.strip() for k in kind_val_str.split(','))
                    break
            filter_order = any(k in ["state_dependency", "operand"] for k in current_kinds_in_ep)
            if filter_order:
                filtered_attributes = [attr for attr in attributes if not attr.startswith("order:")]
            else:
                filtered_attributes = attributes
            return f"EP:{';'.join(sorted(filtered_attributes))}" if filtered_attributes else "EP:edge"
        return original_display_label

    def to_gedgw_input_data(self) -> Dict[str, Any]:
        levi_g = self.to_undirected_levi_graph()
        levi_nodes_with_data = list(levi_g.nodes(data=True))
        node_id_to_int_map = {node_id: i for i, (node_id, _) in enumerate(levi_nodes_with_data)}
        n = levi_g.number_of_nodes()
        m_levi_original = levi_g.number_of_edges()
        all_conceptual_labels = [self.get_conceptual_label_for_node(node_id, data_dict) for node_id, data_dict in levi_nodes_with_data]
        unique_conceptual_labels_list = sorted(list(set(all_conceptual_labels)))
        num_unique_labels = len(unique_conceptual_labels_list)
        conceptual_label_to_idx_map = {label: i for i, label in enumerate(unique_conceptual_labels_list)}
        features_tensor = torch.zeros((n, num_unique_labels), dtype=torch.float)
        for i, (node_id, _) in enumerate(levi_nodes_with_data):
            concept_label = all_conceptual_labels[i]
            features_tensor[i, conceptual_label_to_idx_map[concept_label]] = 1.0
        edge_list_for_tensor = [[node_id_to_int_map[u_levi], node_id_to_int_map[v_levi]] for u_levi, v_levi in levi_g.edges()]
        edge_list_for_tensor_bidirectional = edge_list_for_tensor + [[v, u] for u, v in edge_list_for_tensor]
        self_loops = [[i, i] for i in range(n)]
        # For GEDGW, let's match the paper's likely input if adj is just 0/1, no multi-edges.
        # The self-loops in edge_index are often for GCN/GIN message passing, might not be directly for GED adj matrix.
        # However, models.GEDGW does `self.cost1 = self.g1.adj().to_dense().float()-torch.eye(self.n1)`
        # which implies the input graph for dgl should NOT have self-loops in its edge list if adj() is to be correct.
        # Let's use unique bi-directional edges for now, and dgl will form adj from that.
        # The paper's trainer.py adds self-loops to edge_index for GNN layers.
        # For GEDGW, it uses dgl.graph().adj().
        # So, the edge_list for dgl.graph should be the actual graph edges (bi-directional for undirected).
        
        dgl_edge_list_u = []
        dgl_edge_list_v = []
        for u_int, v_int in edge_list_for_tensor_bidirectional: # Use bi-directional unique edges
             dgl_edge_list_u.append(u_int)
             dgl_edge_list_v.append(v_int)

        # Create a DGL graph to get adjacency matrix without explicit self-loops (as GEDGW subtracts eye(N))
        temp_dgl_g = dgl.graph((torch.tensor(dgl_edge_list_u), torch.tensor(dgl_edge_list_v)), num_nodes=n)
        adj_matrix_for_gedgw = temp_dgl_g.adj().to_dense().float() # This will be C1/C2 in GEDGW

        # The edge_index for models that use GCNConv/GINConv might still need self-loops added later
        # For now, this data structure is for GEDGW which uses adj().
        # The `trainer.py` adds self-loops to edge_index when it packs data.
        # So, provide the bi-directional edges.
        edge_index_tensor_for_gnn = torch.tensor(edge_list_for_tensor_bidirectional + self_loops, dtype=torch.long).t()


        avg_v = float(n)
        hb = float(n + m_levi_original)
        return {
            "n1": n,
            "m1": m_levi_original, # Number of unique undirected edges
            "edge_index_1": edge_index_tensor_for_gnn, # For models expecting GNN input
            "features_1": features_tensor,
            "avg_v": avg_v,
            "hb": hb,
            "ged": 0,
            "target": torch.exp(torch.tensor([0.0 / avg_v if avg_v > 0 else 0.0])).float(),
            "_unique_conceptual_labels_list": unique_conceptual_labels_list,
            # For GEDGW's direct use if it doesn't want GNN style edge_index
            "_adj_matrix_1_for_gedgw": adj_matrix_for_gedgw,
             # Store the original dgl edge list for recreating the DGL graph if needed by GEDGW
            "_dgl_edges_1_u_for_gedgw": dgl_edge_list_u,
            "_dgl_edges_1_v_for_gedgw": dgl_edge_list_v,
        }


def visualize_levi_graph(levi_graph: nx.Graph, title: str = "Levi Graph Visualization", layout_prog: str = 'neato'):
    if not HAS_PYDOT and layout_prog != 'spring':
        print(f"Pydot not found, falling back to 'spring' layout from '{layout_prog}'.")
        layout_prog = 'spring'
    if not levi_graph.nodes:
        print("Graph is empty, cannot visualize.")
        return
    fig, ax = plt.subplots(figsize=(max(15, levi_graph.number_of_nodes() * 0.3), max(12, levi_graph.number_of_nodes() * 0.25)))
    node_color_defs = {
        'Compartment': ('skyblue', NodeRole.COMPARTMENT), 'Exterior': ('grey', NodeRole.EXTERIOR),
        'Operation': ('lightgreen', NodeRole.EXPR_OP), 'Function': ('mediumpurple', NodeRole.EXPR_FUNCTION),
        'Parameter': ('lightcoral', NodeRole.EXPR_SYMBOL, 'param'),
        'Unknown Symbol': ('orange', NodeRole.EXPR_SYMBOL, 'unknown'),
        'Constant': ('lightpink', NodeRole.EXPR_CONSTANT),
        'Unhandled Symbol': ('yellow', NodeRole.EXPR_SYMBOL, 'unhandled'),
        'Edge Proxy': ('lightgrey', NodeRole.EDGE_PROXY)
    }
    node_colors, node_sizes, node_legend_handles = [], [], {}
    base_node_size = 2000
    for n_id_levi in levi_graph.nodes():
        data_levi = levi_graph.nodes[n_id_levi]
        role_levi = data_levi.get('role')
        if not isinstance(role_levi, NodeRole):
            try: role_levi = NodeRole[str(role_levi).split('.')[-1].upper()]
            except KeyError: role_levi = NodeRole.EXTERIOR
        symbol_type_levi = data_levi.get('attrs', {}).get('symbol_type', data_levi.get('symbol_type'))
        color_levi, size_levi, node_type_key_levi = 'white', base_node_size, role_levi
        if role_levi == NodeRole.EXPR_SYMBOL and symbol_type_levi: node_type_key_levi = (role_levi, symbol_type_levi)
        elif role_levi == NodeRole.EDGE_PROXY: size_levi = base_node_size * 0.6
        matched_legend_label_levi = None
        for legend_label_str_levi, definition_levi in node_color_defs.items():
            clr_levi_def, r_def_levi = definition_levi[0], definition_levi[1]
            st_def_levi = definition_levi[2] if len(definition_levi) > 2 else None
            current_def_key_levi = (r_def_levi, st_def_levi) if st_def_levi else r_def_levi
            if node_type_key_levi == current_def_key_levi:
                color_levi, matched_legend_label_levi = clr_levi_def, legend_label_str_levi
                break
        if matched_legend_label_levi and matched_legend_label_levi not in node_legend_handles:
            node_legend_handles[matched_legend_label_levi] = mpatches.Patch(color=color_levi, label=f"Node: {matched_legend_label_levi}")
        elif not matched_legend_label_levi and role_levi and role_levi.name not in node_legend_handles:
            default_label_str_levi = f"Node: {role_levi.name} (Default)"
            node_legend_handles[role_levi.name] = mpatches.Patch(color=color_levi, label=default_label_str_levi)
        node_colors.append(color_levi); node_sizes.append(size_levi)
    labels = {n: levi_graph.nodes[n]['label'] for n in levi_graph.nodes()}
    pos = None
    if layout_prog != 'spring' and HAS_PYDOT:
        try:
            g_for_layout = levi_graph.copy()
            for n_id_layout, node_data_layout in g_for_layout.nodes(data=True):
                if 'label' in node_data_layout and isinstance(node_data_layout['label'], str) and (':' in node_data_layout['label'] or ';' in node_data_layout['label']):
                    g_for_layout.nodes[n_id_layout]['label'] = f'"{node_data_layout["label"]}"'
            pos = graphviz_layout(g_for_layout, prog=layout_prog)
        except Exception as e:
            print(f"Warning: Pydot layout '{layout_prog}' failed ({e}). Using NetworkX spring_layout.")
            pos = nx.spring_layout(levi_graph, k=2.5/np.sqrt(levi_graph.number_of_nodes()) if levi_graph.number_of_nodes() > 0 else 1, iterations=max(70, int(np.log(levi_graph.number_of_nodes() + 1)*30)), seed=42, scale=3.0)
    else:
        if layout_prog != 'spring' and not HAS_PYDOT: print(f"Pydot not found for layout '{layout_prog}'. Using NetworkX spring_layout.")
        pos = nx.spring_layout(levi_graph, k=2.5/np.sqrt(levi_graph.number_of_nodes()) if levi_graph.number_of_nodes() > 0 else 1, iterations=max(70, int(np.log(levi_graph.number_of_nodes() + 1)*30)), seed=42, scale=3.0)
    edge_color_defs = {
        'Source to Proxy': ('teal', 'solid', 1.2, 'source_to_proxy'),
        'Proxy to Target': ('purple', 'solid', 1.2, 'proxy_to_target'),
        'Other Link': ('darkgrey', 'dotted', 0.8, None)
    }
    edge_colors, edge_styles, edge_widths, edge_legend_handles = [], [], [], {}
    for u_edge, v_edge, d_edge_levi in levi_graph.edges(data=True):
        link_type_attr_levi = d_edge_levi.get('link_type')
        current_edge_color, current_edge_style, current_edge_width = edge_color_defs['Other Link'][0:3]
        matched_legend_label_edge_levi = 'Other Link'
        for legend_label_str_edge_levi, definition_edge_levi in edge_color_defs.items():
            clr_edge, sty_edge, wid_edge, lt_def_edge = definition_edge_levi
            if lt_def_edge == link_type_attr_levi:
                current_edge_color, current_edge_style, current_edge_width = clr_edge, sty_edge, wid_edge
                matched_legend_label_edge_levi = legend_label_str_edge_levi; break
        edge_colors.append(current_edge_color); edge_styles.append(current_edge_style); edge_widths.append(current_edge_width)
        if matched_legend_label_edge_levi not in edge_legend_handles:
            edge_legend_handles[matched_legend_label_edge_levi] = mlines.Line2D([], [], color=current_edge_color, linestyle=current_edge_style, linewidth=current_edge_width, label=f"Edge: {matched_legend_label_edge_levi}")
    nx.draw(levi_graph, pos, ax=ax, labels=labels, with_labels=True, node_color=node_colors, node_size=node_sizes, edge_color=edge_colors, style=edge_styles, width=edge_widths, font_size=max(6, int(10 - levi_graph.number_of_nodes()/40.0)), font_weight='normal', arrows=False)
    all_handles_list = sorted(list(node_legend_handles.values()) + list(edge_legend_handles.values()), key=lambda x: x.get_label())
    if all_handles_list: ax.legend(handles=all_handles_list, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='medium', title="Levi Graph Legend", framealpha=0.9)
    ax.set_title(title, fontsize=18); ax.axis('off'); plt.tight_layout(rect=[0, 0, 0.85, 1])
    print(f"Visualization for '{title}' generated (plot display suppressed).")


def gen_edit_path_standalone(data: Dict[str, Any], permute: List[int], unique_conceptual_labels_list: List[str]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], List[str], List[str]]:
    n1 = data["n1"]
    # For self-comparison edge_index_1 and edge_index_2 are derived from the same Levi graph structure
    # but the gen_edit_path needs the GNN-style edge_index (bi-dir + self-loops).
    # The actual edit distance for edges is on the *original* graph structure (no self-loops from GNN prep).
    # So, we need to get the original unique edges from the Levi graph that data['edge_index_1'] was based on.
    # For simplicity in this standalone, we'll assume the GNN edge_index is provided and filter self-loops.
    raw_edges_1_gnn = data["edge_index_1"].t().tolist()
    raw_edges_2_gnn = data["edge_index_2"].t().tolist()

    raw_f1_th = data["features_1"]
    raw_f2_th = data["features_2"]

    assert len(permute) == n1
    assert raw_f1_th.shape[0] == n1 and raw_f2_th.shape[0] == n1
    assert raw_f1_th.shape[1] == raw_f2_th.shape[1]

    edges_1_mapped_to_g2_coords = set()
    for (u_g1_idx, v_g1_idx) in raw_edges_1_gnn:
        if u_g1_idx == v_g1_idx: continue # Skip GNN self-loops for edge comparison
        pu_g2_coord = permute[u_g1_idx]
        pv_g2_coord = permute[v_g1_idx]
        if pu_g2_coord < pv_g2_coord: edges_1_mapped_to_g2_coords.add((pu_g2_coord, pv_g2_coord))
        elif pv_g2_coord < pu_g2_coord: edges_1_mapped_to_g2_coords.add((pv_g2_coord, pu_g2_coord))

    edges_2_g2_coords = set()
    for (u_g2_idx, v_g2_idx) in raw_edges_2_gnn:
        if u_g2_idx == v_g2_idx: continue # Skip GNN self-loops
        if u_g2_idx < v_g2_idx: edges_2_g2_coords.add((u_g2_idx, v_g2_idx))
        elif v_g2_idx < u_g2_idx: edges_2_g2_coords.add((v_g2_idx, u_g2_idx))
            
    edit_edges = edges_1_mapped_to_g2_coords ^ edges_2_g2_coords

    f1_conceptual_indices = [torch.where(raw_f1_th[i] > 0.5)[0][0].item() for i in range(n1)]
    f2_conceptual_indices = [torch.where(raw_f2_th[i] > 0.5)[0][0].item() for i in range(n1)]
    
    relabel_nodes_ops = [] # List of strings describing relabel operations
    relabel_nodes_set = set() # Set of (node_in_g2_coords, new_conceptual_label_idx_from_g1)


    for u_g1_original_idx in range(n1):
        v_g2_mapped_idx = permute[u_g1_original_idx]
        g1_node_concept_idx = f1_conceptual_indices[u_g1_original_idx]
        g2_node_concept_idx = f2_conceptual_indices[v_g2_mapped_idx]
        
        if g1_node_concept_idx != g2_node_concept_idx:
            relabel_nodes_set.add((v_g2_mapped_idx, g1_node_concept_idx))
            original_label_str = unique_conceptual_labels_list[g2_node_concept_idx]
            new_label_str = unique_conceptual_labels_list[g1_node_concept_idx]
            relabel_nodes_ops.append(f"Relabel node {v_g2_mapped_idx} from '{original_label_str}' to '{new_label_str}'")

    edit_edges_ops = []
    for edge in edit_edges:
        if edge in edges_1_mapped_to_g2_coords and edge not in edges_2_g2_coords:
            edit_edges_ops.append(f"Delete edge {edge} (mapped from G1)")
        elif edge in edges_2_g2_coords and edge not in edges_1_mapped_to_g2_coords:
             edit_edges_ops.append(f"Insert edge {edge} (present in G2)")


    return edit_edges_ops, relabel_nodes_ops # Return lists of strings for printing


# --- Example Usage ---
if __name__ == "__main__":
    state_vars_ex = ['S', 'E', 'A', 'Q', 'I', 'R']
    params_ex = [
        'Lambda', 'alpha', 'beta', 'gamma1', 'gamma2', 'gamma3',
        'delta1', 'mu1', 'gamma4', 'delta2', 'mu2', 'sigma1',
        'a', 'b', 'd', 'epsilon', 'rho', 'omega'
    ]

    # IMPORTANT: Ensure the sympy symbols used in expressions are the ones
    # from the EpiModelGraph's internal self.param_symbol_objects if you want
    # them to be recognized as parameters.
    # For this example, we'll rely on the graph's internal creation.
    # The global sympy symbols are just for constructing expressions passed to add_interaction.
    S_sym, E_sym, A_sym, Q_sym, I_sym, R_sym = sp.symbols(state_vars_ex)
    Lambda_sym, alpha_sym, beta_sym, gamma1_sym, gamma2_sym, gamma3_sym, \
    delta1_sym, mu1_sym, gamma4_sym, delta2_sym, mu2_sym, sigma1_sym, \
    a_sym, b_sym, d_sym, epsilon_sym, rho_sym, omega_sym = sp.symbols(params_ex)


    graph_ex = EpiModelGraph(
        name="COVID-19_Spain_Treatment",
        state_vars=state_vars_ex,
        params=params_ex
    )
    graph_ex.add_interaction(source=EpiModelGraph.EXTERIOR_NODE_LABEL, target='S', expr=Lambda_sym)
    infection_expr_ex = (alpha_sym*A_sym + beta_sym*I_sym) * S_sym
    graph_ex.add_interaction(source='S', target='E', expr=infection_expr_ex)
    graph_ex.add_interaction(source='S', target=EpiModelGraph.EXTERIOR_NODE_LABEL, expr=epsilon_sym * S_sym)
    graph_ex.add_interaction(source='E', target=EpiModelGraph.EXTERIOR_NODE_LABEL, expr=epsilon_sym * E_sym)
    graph_ex.add_interaction(source='A', target=EpiModelGraph.EXTERIOR_NODE_LABEL, expr=epsilon_sym * A_sym)
    graph_ex.add_interaction(source='Q', target=EpiModelGraph.EXTERIOR_NODE_LABEL, expr=epsilon_sym * Q_sym)
    graph_ex.add_interaction(source='I', target=EpiModelGraph.EXTERIOR_NODE_LABEL, expr=epsilon_sym * I_sym)
    graph_ex.add_interaction(source='R', target=EpiModelGraph.EXTERIOR_NODE_LABEL, expr=epsilon_sym * R_sym)
    graph_ex.add_interaction(source='E', target='A', expr=gamma1_sym * E_sym)
    graph_ex.add_interaction(source='E', target='Q', expr=gamma2_sym * E_sym)
    graph_ex.add_interaction(source='E', target='I', expr=gamma3_sym * E_sym)
    graph_ex.add_interaction(source='E', target='R', expr=gamma4_sym * E_sym)
    graph_ex.add_interaction(source='A', target='I', expr=delta1_sym * A_sym)
    graph_ex.add_interaction(source='A', target='R', expr=delta2_sym * A_sym)
    graph_ex.add_interaction(source='Q', target='I', expr=mu1_sym * Q_sym)
    graph_ex.add_interaction(source='Q', target='R', expr=mu2_sym * Q_sym)
    graph_ex.add_interaction(source='I', target='R', expr=sigma1_sym * I_sym)
    treatment_expr_ex = (a_sym / (sp.Integer(1) + b_sym*I_sym)) * I_sym
    graph_ex.add_interaction(source='I', target='R', expr=treatment_expr_ex)
    graph_ex.add_interaction(source='I', target=EpiModelGraph.EXTERIOR_NODE_LABEL, expr=d_sym * I_sym)
    tau_sym = sp.symbols('tau')
    integral_expr_ex = sp.Integral(rho_sym * E_sym, (tau_sym, 0, omega_sym))
    graph_ex.add_interaction(source='E', target='R', expr=integral_expr_ex)

    print("--- EpiModelGraph Summary ---")
    print(graph_ex.summary())

    print("\n--- GEDGW Self-Comparison Test ---")
    gedgw_single_graph_data_ex = graph_ex.to_gedgw_input_data()

    # For GEDGW, construct the data dict as it expects it for G1 vs G2
    # Here G1 and G2 are the same graph.
    data_for_gedgw_self_comparison_ex = {
        "n1": gedgw_single_graph_data_ex["n1"],
        "n2": gedgw_single_graph_data_ex["n1"], # n2 is same as n1
        # GEDGW uses DGL graphs built from edge lists, not the GNN edge_index directly for adj.
        # It constructs g1 and g2 from edge_index_1[0], edge_index_1[1] etc.
        # The trainer.py packs data by providing edge_index_1, features_1, etc.
        # And GEDGW's __init__ creates DGL graphs from these.
        # The edge_index should be the one suitable for creating the DGL graph (bi-directional for undirected)
        "edge_index_1": torch.stack([torch.tensor(gedgw_single_graph_data_ex["_dgl_edges_1_u_for_gedgw"]),
                                     torch.tensor(gedgw_single_graph_data_ex["_dgl_edges_1_v_for_gedgw"])]),
        "edge_index_2": torch.stack([torch.tensor(gedgw_single_graph_data_ex["_dgl_edges_1_u_for_gedgw"]),
                                     torch.tensor(gedgw_single_graph_data_ex["_dgl_edges_1_v_for_gedgw"])]),
        "features_1": gedgw_single_graph_data_ex["features_1"],
        "features_2": gedgw_single_graph_data_ex["features_1"], # Same features
        # m1 and m2 are original edge counts, not tensor edge counts
        "m1": gedgw_single_graph_data_ex["m1"],
        "m2": gedgw_single_graph_data_ex["m1"],
        "avg_v": gedgw_single_graph_data_ex["avg_v"],
        "hb": gedgw_single_graph_data_ex["hb"]
        # Other fields like 'ged', 'target' are not strictly used by GEDGW.process but good for consistency
    }
    
    num_levi_nodes_ex = data_for_gedgw_self_comparison_ex['n1']
    print(f"Levi graph for GEDGW has {num_levi_nodes_ex} nodes.")
    print(f"Original unique edges in Levi graph: {data_for_gedgw_self_comparison_ex['m1']}.")

    class DummyArgs: # Mimic the args object expected by GEDGW
        dataset = "CUSTOM_EPI_MODEL"
    args_ex = DummyArgs()

    try:
        gedgw_model_instance_ex = GEDGW(data_for_gedgw_self_comparison_ex, args_ex)
        T_cg_ex, predicted_ged_value_ex = gedgw_model_instance_ex.process()
        print(f"\nGEDGW Predicted GED (self-comparison): {predicted_ged_value_ex}")
        if abs(predicted_ged_value_ex) < 1e-5 :
             print("Predicted GED is close to 0, as expected for self-comparison.")
        else:
             print(f"WARNING: Predicted GED is {predicted_ged_value_ex}, not close to 0 for self-comparison.")

        print("\n--- Edit Path Generation (Self-Comparison with Identity Mapping) ---")
        identity_permutation_ex = list(range(num_levi_nodes_ex))
        
        # Data for gen_edit_path_standalone needs the GNN-style edge_index
        data_for_edit_path_check = {
            "n1": gedgw_single_graph_data_ex["n1"],
            "edge_index_1": gedgw_single_graph_data_ex["edge_index_1"], # GNN style
            "edge_index_2": gedgw_single_graph_data_ex["edge_index_1"], # GNN style
            "features_1": gedgw_single_graph_data_ex["features_1"],
            "features_2": gedgw_single_graph_data_ex["features_1"],
        }
        unique_labels_list = gedgw_single_graph_data_ex["_unique_conceptual_labels_list"]
        
        edit_edge_ops_ex, relabel_node_ops_ex = gen_edit_path_standalone(
            data_for_edit_path_check, 
            identity_permutation_ex,
            unique_labels_list
        )

        print(f"Number of edit edge operations (for identity mapping): {len(edit_edge_ops_ex)}")
        if edit_edge_ops_ex:
            for op in edit_edge_ops_ex: print(f"  {op}")
        
        print(f"Number of relabeling operations (for identity mapping): {len(relabel_node_ops_ex)}")
        if relabel_node_ops_ex:
            for op in relabel_node_ops_ex: print(f"  {op}")
        
        calculated_ged_from_identity_ex = len(edit_edge_ops_ex) + len(relabel_node_ops_ex)
        print(f"Calculated GED from identity path: {calculated_ged_from_identity_ex}")
        if calculated_ged_from_identity_ex == 0:
            print("Edit path is empty for identity mapping. SUCCESS: Graph is identical to itself based on conceptual labels.")
        else:
            print("WARNING: Edit path for identity mapping is NOT empty.")

    except ImportError as e:
        print(f"ImportError: {e}. Ensure libraries (torch, dgl, pot) and src modules are accessible.")
    except AttributeError as e:
        print(f"AttributeError: {e}. Check DummyArgs or model setup.")
        import traceback; traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error in GEDGW test: {type(e).__name__} - {e}")
        import traceback; traceback.print_exc()