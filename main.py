import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Set, Any

MAX_NODES = 16

BASE_NODE_SIZE = 260


EXAMPLE_TREES = {
    "Perfect (height 3)": ["A", "B", "C", "D", "E", "F", "G"],
    "Left-skewed (height 4)": ["A", "B", None, "C", None, None, None, "D"],
    "Right-skewed (height 4)": ["A", None, "B", None, None, None, "C"],
    "Complete but not full": ["A", "B", "C", "D", "E", "F", None],
}

st.set_page_config(
    page_title="Discrete Structures Project",
    layout="wide",
)

plt.style.use("dark_background")


# ----------------- Graph Helpers (Page 1) ----------------- #

def parse_edge_list(
    text: str,
    directed: bool = False,
    weighted: bool = False,
    max_nodes: int = MAX_NODES,
) -> Tuple[Optional[nx.Graph], Optional[str]]:
    """Parse multiline edge list into a Graph / DiGraph."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None, "Please enter at least one non-empty line of edges."

    G = nx.DiGraph() if directed else nx.Graph()
    nodes: Set[Any] = set()

    for i, line in enumerate(lines, start=1):
        parts = line.split()

        if weighted:
            if len(parts) != 3:
                return None, f"Line {i}: expected 'u v w', got: {line!r}"
            u, v, w_str = parts
            try:
                w = float(w_str)
            except ValueError:
                return None, f"Line {i}: weight {w_str!r} is not a valid number."
            G.add_edge(u, v, weight=w)
            nodes.update([u, v])
        else:
            if len(parts) != 2:
                return None, f"Line {i}: expected 'u v', got: {line!r}"
            u, v = parts
            G.add_edge(u, v)
            nodes.update([u, v])

        if len(nodes) > max_nodes:
            return None, f"Node limit exceeded: maximum {max_nodes} distinct nodes are allowed."

    if G.number_of_nodes() == 0:
        return None, "Parsed graph has no nodes; please check your input."

    return G, None


def draw_graph(
    G: nx.Graph,
    title: str = "",
    highlight_edges: Optional[List[tuple]] = None,
    highlight_nodes: Optional[List[Any]] = None,
    fig_size: tuple = (7.0, 3.5),
):
    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#121212")

    if G.number_of_nodes() == 0:
        ax.set_title("Empty graph", color="white")
        ax.axis("off")
        return fig

    pos = choose_layout(G)

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color="#008cff",
        edgecolors="white",
        linewidths=1.5,
        node_size=BASE_NODE_SIZE,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_color="white",
        font_size=10,
    )

    base_edge_color = "#cccccc"
    if G.is_directed():
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color=base_edge_color,
            arrows=True,
            arrowstyle="->",
            arrowsize=13,
            width=1.6,
        )
    else:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color=base_edge_color,
            width=1.8,
        )

    if highlight_edges:
        if G.is_directed():
            he_set = {(u, v) for (u, v) in highlight_edges}
        else:
            he_set = {tuple(sorted((u, v))) for (u, v) in highlight_edges}
        for (u, v) in G.edges():
            key = (u, v) if G.is_directed() else tuple(sorted((u, v)))
            if key in he_set:
                if G.is_directed():
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=[(u, v)],
                        ax=ax,
                        edge_color="#ff5252",
                        width=3,
                        arrows=True,
                        arrowstyle="->",
                        arrowsize=16,
                    )
                else:
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=[(u, v)],
                        ax=ax,
                        edge_color="#ff5252",
                        width=3,
                    )

    if highlight_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=highlight_nodes,
            ax=ax,
            node_color="#4caf50",
            edgecolors="white",
            linewidths=2,
            node_size=BASE_NODE_SIZE,
        )

    ax.set_title(title, color="white", pad=8)
    ax.axis("off")
    return fig

def choose_layout(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()

    degrees = [deg for _, deg in G.degree]
    max_deg = max(degrees) if degrees else 0

    is_path_like = (m == n - 1) and (max_deg <= 2)

    if is_path_like and n > 2:

        return nx.circular_layout(G)
    else:
        return nx.spring_layout(G, seed=42, k=1.2, scale=3.0)

# ---------- Tree helpers (Page 2) ----------

def parse_level_order_list(text: str) -> List[Optional[str]]:
    """
    Parse a single line like: A B C D E None F
    into a Python list representing level-order nodes.
    Use None / - to mean 'no node' at that position.
    """
    tokens = [t.strip() for t in text.split() if t.strip()]
    if not tokens:
        return []

    result: List[Optional[str]] = []
    for tok in tokens:
        if tok.lower() in {"none", "null", "-"}:
            result.append(None)
        else:
            result.append(tok)
    return result


def build_binary_tree_edges(level_list: List[Optional[str]]) -> List[tuple]:
    """
    Given a level-order list, return edges (parent, child) for non-None entries.
    Indices follow heap-style: left = 2*i+1, right = 2*i+2.
    """
    edges: List[tuple] = []
    n = len(level_list)
    for i, val in enumerate(level_list):
        if val is None:
            continue
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and level_list[left] is not None:
            edges.append((val, level_list[left]))
        if right < n and level_list[right] is not None:
            edges.append((val, level_list[right]))
    return edges


def binary_tree_traversals(level_list: List[Optional[str]]):
    """
    Return dict with preorder, inorder, postorder, levelorder lists of node labels.
    """
    # map index -> value
    def preorder(i: int, out: List[str]):
        if i >= len(level_list) or level_list[i] is None:
            return
        out.append(level_list[i])
        preorder(2 * i + 1, out)
        preorder(2 * i + 2, out)

    def inorder(i: int, out: List[str]):
        if i >= len(level_list) or level_list[i] is None:
            return
        inorder(2 * i + 1, out)
        out.append(level_list[i])
        inorder(2 * i + 2, out)

    def postorder(i: int, out: List[str]):
        if i >= len(level_list) or level_list[i] is None:
            return
        postorder(2 * i + 1, out)
        postorder(2 * i + 2, out)
        out.append(level_list[i])

    def levelorder(out: List[str]):
        for val in level_list:
            if val is not None:
                out.append(val)

    result = {}
    if not level_list or level_list[0] is None:
        return {"preorder": [], "inorder": [], "postorder": [], "levelorder": []}

    pre, ino, post, lev = [], [], [], []
    preorder(0, pre)
    inorder(0, ino)
    postorder(0, post)
    levelorder(lev)
    return {"preorder": pre, "inorder": ino, "postorder": post, "levelorder": lev}


def binary_tree_properties(level_list: List[Optional[str]]):
    """Return height, leaf count, internal count, and simple full/complete flags."""
    if not level_list or level_list[0] is None:
        return {"height": 0, "leaves": 0, "internals": 0, "is_full": False, "is_complete": False}

    # height: max depth of non-None index
    max_idx = max(i for i, v in enumerate(level_list) if v is not None)
    height = max_idx.bit_length()  # floor(log2(max_idx)) + 1

    leaves = 0
    internals = 0
    for i, v in enumerate(level_list):
        if v is None:
            continue
        left = 2 * i + 1
        right = 2 * i + 2
        has_left = left < len(level_list) and level_list[left] is not None
        has_right = right < len(level_list) and level_list[right] is not None
        if not has_left and not has_right:
            leaves += 1
        else:
            internals += 1


    is_full = all(
        v is None or (
            ((2 * i + 1) < len(level_list) and level_list[2 * i + 1] is not None) and
            ((2 * i + 2) < len(level_list) and level_list[2 * i + 2] is not None)
        ) or (
            ((2 * i + 1) >= len(level_list) or level_list[2 * i + 1] is None) and
            ((2 * i + 2) >= len(level_list) or level_list[2 * i + 2] is None)
        )
        for i, v in enumerate(level_list) if v is not None
    )

    # complete: ignore trailing Nones at end
    from itertools import dropwhile
    trimmed = list(dropwhile(lambda x: x is None, level_list))
    is_complete = None not in trimmed[:-1]

    return {
        "height": height,
        "leaves": leaves,
        "internals": internals,
        "is_full": is_full,
        "is_complete": is_complete,
    }

def draw_binary_tree(level_list: List[Optional[str]], title: str = "Binary tree"):
    """Draw a binary tree using a layered layout."""
    import math

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#121212")

    if not level_list or level_list[0] is None:
        ax.set_title("Empty tree", color="white")
        ax.axis("off")
        return fig

    # compute depth for each node
    positions = {}
    n = len(level_list)
    max_idx = max(i for i, v in enumerate(level_list) if v is not None)
    height = max_idx.bit_length()

    # horizontal spacing per depth
    for i, val in enumerate(level_list):
        if val is None:
            continue
        depth = int(math.floor(math.log2(i + 1)))
        # position within that depth
        idx_in_level = i - (2 ** depth - 1)
        slots = 2 ** depth
        x = (idx_in_level + 1) / (slots + 1)
        y = 1.0 - depth / max(height, 1.0)
        positions[val] = (x, y)

    edges = build_binary_tree_edges(level_list)
    G = nx.DiGraph()
    for val in [v for v in level_list if v is not None]:
        G.add_node(val)
    G.add_edges_from(edges)

    nx.draw_networkx_nodes(
        G, positions, ax=ax,
        node_color="#008cff", edgecolors="white",
        linewidths=1.5, node_size=320,
    )
    nx.draw_networkx_labels(G, positions, ax=ax, font_color="white", font_size=10)
    nx.draw_networkx_edges(
        G, positions, ax=ax, edge_color="#cccccc",
        arrows=False, width=1.8,
    )

    ax.set_title(title, color="white", pad=8)
    ax.axis("off")
    return fig

def validate_binary_tree(level_list: List[Optional[str]]) -> Optional[str]:
    """Return error message string if invalid, else None."""
    if not level_list:
        return "Tree is empty."

    if level_list[0] is None:
        return "Root must not be empty."

    # build adjacency as an undirected graph for connectivity / cycle checks
    edges = build_binary_tree_edges(level_list)
    labels = [v for v in level_list if v is not None]

    # 1) detect multiple components (disconnected)
    # run BFS from root and see if all labels are reachable
    from collections import deque

    adj = {v: [] for v in labels}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    root = level_list[0]
    seen = set([root])
    q = deque([root])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                q.append(w)
    if seen != set(labels):
        return "Invalid tree: level-order contains disconnected nodes (forest instead of a single tree)."

    # 2) detect cycles (including self-loops / repeated parent-child edges)
    # standard DFS cycle check on undirected graph
    parent = {}

    def dfs(u, p):
        for w in adj[u]:
            if w == p:
                continue
            if w in parent:
                return True
            parent[w] = u
            if dfs(w, u):
                return True
        return False

    parent[root] = None
    if dfs(root, None):
        return "Invalid tree: cycle detected in the structure."

    return None

def list_to_text(level_list: List[Optional[str]]) -> str:
    """Convert a level-order list into a space-separated string."""
    return " ".join(str(x) if x is not None else "None" for x in level_list)

# ---------- MST helpers (Page 3) ----------

def is_tree_undirected(G: nx.Graph) -> Tuple[bool, str]:
    """Return (is_tree, explanation) for an undirected graph."""
    if G.number_of_nodes() == 0:
        return False, "Graph has `no` nodes."
    if not nx.is_connected(G):
        return False, "Graph is `not` connected, so it **cannot** be a tree."
    m = G.number_of_edges()
    n = G.number_of_nodes()
    if m != n - 1:
        return False, f"Connected but has `{m}` edges and `{n}` nodes; a tree must have exactly `n-1` edges."
    return True, "Graph is connected and has m = n - 1 edges, so it is a tree."

# ----------------- Page 1: Basic Graphs ----------------- #

def page_simple_graphs():
    st.header("Page 1: Simple Graphs & Traversals")

    st.markdown(
        """
Here you can build a small **simple graph** (≤ 16 nodes) and try:

- Breadth-first search (BFS) and depth-first search (DFS).
- See connected components grouped by color (for undirected graphs).

#### To use:

1. Enter an unweighted edge list
2. Build the graph
3. Use the various other tools
---
"""
    )

    # ========== ROW 1: Graph setup | Graph view ========== #
    left_top, right_top = st.columns([1.1, 1.3])

    # ---- Left: setup ---- #
    with left_top:
        st.subheader("Graph setup")

        graph_type = st.radio("Graph type", ["Undirected", "Directed"], horizontal=True)
        directed = graph_type == "Directed"

        st.caption("Edge list (one edge per line, format: `u v`, where `u` and `v` are node names):")

        example_text = "A B\nB C\nC D\nA D\nB D"
        edge_text = st.text_area(
            "Edge list input",
            value=example_text,
            height=140,
            label_visibility="collapsed",
        )

        build_clicked = st.button("Build / Update graph", use_container_width=True)

        if build_clicked:
            G, err = parse_edge_list(edge_text, directed=directed, weighted=False)
            if err:
                st.error(err)
                st.session_state["page1_graph"] = None
            else:
                st.session_state["page1_graph"] = (G, directed)
                # reset previous results
                st.session_state["page1_bfs_order"] = None
                st.session_state["page1_bfs_edges"] = None
                st.session_state["page1_dfs_order"] = None
                st.session_state["page1_dfs_edges"] = None
                st.session_state["page1_components"] = None
                st.session_state["page1_active_traversal"] = None

    # ---- Right: graph view ---- #
    with right_top:
        st.subheader("Graph view")
        stored = st.session_state.get("page1_graph", None)
        if stored:
            G, directed = stored
            # Fixed-size figure so it visually matches setup height
            fig = draw_graph(
                G,
                title=f"{'Directed' if directed else 'Undirected'} graph",
                fig_size=(7.0, 3.5),   # tweak if you want a bit taller/shorter
            )
            st.pyplot(fig)
            st.caption(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
        else:
            st.info("Define a graph on the left and click **Build / Update graph**.")

    st.markdown("---")

    # ---------- ROW 2: Algorithms | Traversal results ----------
    left_alg, right_alg = st.columns([1.1, 1.3])

    with left_alg:
        st.subheader("Algorithms")
        stored = st.session_state.get("page1_graph", None)
        if not stored:
            st.info("Build a graph above to enable algorithms.")
        else:
            G, directed = stored
            if G.number_of_nodes() == 0:
                st.info("Graph has no nodes.")
            else:
                start_node = st.selectbox(
                    "Start node for BFS/DFS:",
                    options=list(G.nodes()),
                    key="page1_start_node",
                )
                col_bfs, col_dfs = st.columns(2)
                with col_bfs:
                    bfs_clicked = st.button("Run BFS", use_container_width=True, key="page1_bfs_btn")
                with col_dfs:
                    dfs_clicked = st.button("Run DFS", use_container_width=True, key="page1_dfs_btn")

                if bfs_clicked:
                    try:
                        bfs_tree = nx.bfs_tree(G, start_node)
                        st.session_state["page1_active_traversal"] = "bfs"
                        st.session_state["page1_bfs_order"] = list(bfs_tree.nodes())
                        st.session_state["page1_bfs_edges"] = list(bfs_tree.edges())
                        # clear DFS state so only BFS is considered
                        st.session_state["page1_dfs_order"] = None
                        st.session_state["page1_dfs_edges"] = None
                    except Exception as e:
                        st.error(f"BFS failed: {e}")
                        st.session_state["page1_bfs_order"] = None
                        st.session_state["page1_bfs_edges"] = None

                if dfs_clicked:
                    try:
                        dfs_tree = nx.dfs_tree(G, start_node)
                        st.session_state["page1_active_traversal"] = "dfs"
                        st.session_state["page1_dfs_order"] = list(dfs_tree.nodes())
                        st.session_state["page1_dfs_edges"] = list(dfs_tree.edges())
                        # clear BFS state so only DFS is considered
                        st.session_state["page1_bfs_order"] = None
                        st.session_state["page1_bfs_edges"] = None
                    except Exception as e:
                        st.error(f"DFS failed: {e}")
                        st.session_state["page1_dfs_order"] = None
                        st.session_state["page1_dfs_edges"] = None


    with right_alg:
        st.subheader("Traversal results")

        stored = st.session_state.get("page1_graph")
        if not stored:
            st.info("Run BFS/DFS on the left to see results.")
            return

        G, directed = stored
        if G.number_of_nodes() == 0:
            st.info("Graph has no nodes yet.")
            return

        active = st.session_state.get("page1_active_traversal")
        bfs_order = st.session_state.get("page1_bfs_order")
        dfs_order = st.session_state.get("page1_dfs_order")

        if active == "bfs" and bfs_order:
            st.markdown("**BFS visit order:** " + " → ".join(map(str, bfs_order)))
            bfs_edges = st.session_state.get("page1_bfs_edges") or []
            fig_bfs = draw_graph(
                G,
                title="BFS tree (highlighted edges)",
                fig_size=(7.0, 3.5),
                highlight_edges=bfs_edges,
            )
            st.pyplot(fig_bfs)

        elif active == "dfs" and dfs_order:
            st.markdown("**DFS visit order:** " + " → ".join(map(str, dfs_order)))
            dfs_edges = st.session_state.get("page1_dfs_edges") or []
            fig_dfs = draw_graph(
                G,
                title="DFS tree (highlighted edges)",
                fig_size=(7.0, 3.5),
                highlight_edges=dfs_edges,
            )
            st.pyplot(fig_dfs)

        else:
            st.info("Click **Run BFS** or **Run DFS** to view a traversal.")

    st.markdown("---")

    # ---------- ROW 3: Components | Components graph ----------
    left_cc, right_cc = st.columns([1.1, 1.3])

    with left_cc:
        st.subheader("Connected components (undirected only)")
        stored = st.session_state.get("page1_graph", None)
        if not stored:
            st.info("Build a graph above first.")
        else:
            G, directed = stored
            if directed:
                st.caption(
                    "Components here are defined for undirected graphs. "
                    "Switch to *Undirected* to view them."
                )
            elif G.number_of_nodes() == 0:
                st.info("Graph has no nodes.")
            else:
                cc_clicked = st.button(
                    "Show components", use_container_width=True, key="page1_cc_btn"
                )
                if cc_clicked:
                    try:
                        comps = list(nx.connected_components(G))
                        st.session_state["page1_components"] = comps
                    except Exception as e:
                        st.error(f"Failed to compute components: {e}")
                        st.session_state["page1_components"] = None

    with right_cc:
        st.subheader("Components view")
        stored = st.session_state.get("page1_graph", None)
        comps = st.session_state.get("page1_components")
        if not stored or comps is None:
            st.info("Click **Show components** on the left to see the visualization.")
        else:
            G, directed = stored
            if G.number_of_nodes() > 0 and not directed:
                pos = choose_layout(G)
                fig_cc, ax = plt.subplots(figsize=(7.0, 3.5))
                fig_cc.patch.set_facecolor("#121212")
                ax.set_facecolor("#121212")

                colors = ["#008cff", "#ad6800", "#008304", "#84009b","#9c492f", "#3b6a70", "#993c5b", "#707a15"]
                for i, comp in enumerate(comps):
                    nx.draw_networkx_nodes(
                        G,
                        pos,
                        nodelist=list(comp),
                        node_color=colors[i % len(colors)],
                        edgecolors="white",
                        linewidths=1.5,
                        node_size=BASE_NODE_SIZE,
                        ax=ax,
                    )
                nx.draw_networkx_labels(G, pos, ax=ax, font_color="white")
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc", width=1.5)
                ax.set_title("Connected components (color-coded)", color="white", pad=8)
                ax.axis("off")
                st.pyplot(fig_cc)

# ----------------- Page 2 (trees) ----------------- #

def page_trees():
    st.header("Page 2: Rooted & Binary Trees")

    st.markdown(
        """
Here you can make **binary trees** and view their different traversals.

#### To use:

1. Choose a preset example from the dropdown provided.
2. Or make/edit your *own* tree using level-order sequence in the textbox.

---
"""
    )

    if "page2_level_list" not in st.session_state:
        st.session_state["page2_level_list"] = None

    # ---------- Row 1: all inputs | tree view ----------
    left_top, right_top = st.columns([1.2, 1.3])

    with left_top:
        st.subheader("Tree input")

        # --- 1) Preset examples ---
        ex_col1, ex_col2 = st.columns([2, 1])
        with ex_col1:
            example_name = st.selectbox(
                "Preset example",
                options=[
                    "Use custom input",
                    "Perfect (height 3)",
                    "Left-skewed (height 4)",
                    "Right-skewed (height 4)",
                    "Complete but not full",
                ],
                index=0,
            )

        # INITIALIZE / UPDATE RAW TEXT IN STATE *BEFORE* TEXTAREA
        if "page2_level_text" not in st.session_state:
            st.session_state.page2_level_text = "A B C D E F G"

        # If a preset is chosen, update state text (safe: still before textarea)
        if example_name != "Use custom input":
            preset_list = EXAMPLE_TREES[example_name]
            st.session_state.page2_level_text = list_to_text(preset_list)

        st.caption("Level-order nodes (use None / - for missing children):")
        level_text = st.text_area(
            "Level-order sequence",
            height=80,
            key="page2_level_text",
        )

        # Parse textarea on every rerun
        raw_list = parse_level_order_list(level_text.strip())

        build_tree_clicked = st.button("Build / Update tree", use_container_width=True, key="page2_build_btn")

        if build_tree_clicked:
            level_list = raw_list

            if len(level_list) > 31:
                st.error("Please keep the tree reasonably small (≤ 31 positions in level order).")
                level_list = level_list[:31]

            err = validate_binary_tree(level_list)
            if err is not None:
                st.error(err)
                st.session_state.page2_level_list = None
            else:
                st.session_state.page2_level_list = level_list


    with right_top:
        st.subheader("Tree visualization")
        level_list = st.session_state.get("page2_level_list")
        if not level_list:
            st.info(
                "Choose a preset, edit the level-order text, or set a root at index 0 "
                "to build a tree."
            )
        else:
            fig = draw_binary_tree(level_list, title="Binary tree")
            st.pyplot(fig)

    st.markdown("---")

    # ---------- Row 2: Traversals & properties ----------
    left_bottom, right_bottom = st.columns([1.1, 1.3])

    with left_bottom:
        st.subheader("Traversals")
        level_list = st.session_state.get("page2_level_list")
        if not level_list or level_list[0] is None:
            st.info("Build a tree above to see traversals.")
        else:
            trav = binary_tree_traversals(level_list)
            st.markdown("**Preorder:** " + " → ".join(trav["preorder"]))
            st.markdown("**Inorder:** " + " → ".join(trav["inorder"]))
            st.markdown("**Postorder:** " + " → ".join(trav["postorder"]))
            st.markdown("**Level-order:** " + " → ".join(trav["levelorder"]))

    with right_bottom:
        st.subheader("Properties")
        level_list = st.session_state.get("page2_level_list")
        if not level_list or level_list[0] is None:
            st.info("Build a tree above to view properties.")
        else:
            props = binary_tree_properties(level_list)
            st.write(f"Height: `{props['height']}`")
            st.write(f"Leaves: `{props['leaves']}`")
            st.write(f"Internal nodes: `{props['internals']}`")
            st.write(f"Full binary tree: `{'Yes' if props['is_full'] else 'No'}`")
            st.write(f"Complete binary tree: `{'Yes' if props['is_complete'] else 'No'}`")

# ----------------- Page 3: Spanning Trees & MSTs ----------------- #
def page_spanning_trees():
    st.header("Page 3: Spanning Trees & MSTs")

    st.markdown(
        """
Here you can generate **minimum spanning trees (MSTs)** for weighted, undirected graphs, as well as the Kruskal algorithm for minimal weight sum.

> We omitted prim's algorithm as it would usually visually result in a very similar looking graph.

#### To use:
1. Enter a weighted edge list.
2. Build the graph.
3. Use the buttons to switch between the original graph, a spanning tree, and an MST view.
---
"""
    )

    if "page3_graph" not in st.session_state:
        st.session_state["page3_graph"] = None
    if "page3_mst_edges" not in st.session_state:
        st.session_state["page3_mst_edges"] = None
    if "page3_spanning_tree_edges" not in st.session_state:
        st.session_state["page3_spanning_tree_edges"] = None
    if "page3_view_mode" not in st.session_state:
        st.session_state["page3_view_mode"] = "base"  # "base", "span", "mst"

    # ---------- Row 1: controls only (no graph) ----------
    left_controls, right_controls = st.columns([1.1, 1.3])

    with left_controls:
        st.subheader("Weighted graph input")

        st.caption("Undirected, weighted edges (one per line, format: `u v w` where `u` and `v` are node names, `w` is the weight of edge):")
        example_w = "a b 4\na c 2\nb c 5\nb d 10\nc d 3"
        edge_text = st.text_area(
            "Weighted edge list",
            value=example_w,
            height=150,
            key="page3_edge_text",
        )

        build_clicked = st.button(
            "Build / Update weighted graph",
            use_container_width=True,
            key="page3_build_btn",
        )

        if build_clicked:
            G, err = parse_edge_list(
                edge_text,
                directed=False,
                weighted=True,
                max_nodes=MAX_NODES,
            )
            if err:
                st.error(err)
                st.session_state["page3_graph"] = None
                st.session_state["page3_mst_edges"] = None
                st.session_state["page3_spanning_tree_edges"] = None
                st.session_state["page3_view_mode"] = "base"
            else:
                st.session_state["page3_graph"] = G
                st.session_state["page3_mst_edges"] = None
                st.session_state["page3_spanning_tree_edges"] = None
                st.session_state["page3_view_mode"] = "base"

    with right_controls:
        st.subheader("Tree & MST tools")

        G = st.session_state.get("page3_graph")
        if not G:
            st.info("Build a weighted graph on the left to enable the tools.")
        elif G.number_of_nodes() == 0:
            st.info("Graph has no nodes.")
        else:
            is_tree, reason = is_tree_undirected(G)
            st.write(f"**Is the graph a tree?** {'`Yes`' if is_tree else '`No`'}")
            st.markdown("> " + reason)

            st.markdown("##### View:")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                base_btn = st.button("Original graph", use_container_width=True, key="page3_view_base")
            with col_b:
                span_btn = st.button("One spanning tree", use_container_width=True, key="page3_view_span")
            with col_c:
                mst_btn = st.button("MST (Kruskal)", use_container_width=True, key="page3_view_mst")

            if base_btn:
                st.session_state["page3_view_mode"] = "base"

            if span_btn:
                try:
                    root = next(iter(G.nodes()))
                    T = nx.bfs_tree(G, root)
                    st.session_state["page3_spanning_tree_edges"] = list(T.edges())
                    st.session_state["page3_view_mode"] = "span"
                except Exception as e:
                    st.error(f"Failed to build spanning tree: {e}")
                    st.session_state["page3_spanning_tree_edges"] = None

            if mst_btn:
                try:
                    T_mst = nx.minimum_spanning_tree(G, algorithm="kruskal")
                    st.session_state["page3_mst_edges"] = list(T_mst.edges(data=True))
                    st.session_state["page3_view_mode"] = "mst"
                except Exception as e:
                    st.error(f"Failed to compute MST: {e}")
                    st.session_state["page3_mst_edges"] = None

    st.markdown("---")

    # ---------- Row 2: graph view ----------
    G = st.session_state.get("page3_graph")
    if not G:
        st.info("Build a weighted graph above to see the visualization.")
        return
    if G.number_of_nodes() == 0:
        st.info("Graph has no nodes.")
        return

    # center header + graph in a narrower middle column
    left_pad, center_col, right_pad = st.columns([1, 2, 1])
    with center_col:
        st.markdown("### <div style='text-align:center'>Graph view</div>", unsafe_allow_html=True)

        mode = st.session_state.get("page3_view_mode", "base")
        highlight_edges = []
        title = "Weighted graph (edge labels = weights)"

        if mode == "span":
            highlight_edges = st.session_state.get("page3_spanning_tree_edges") or []
            title = "One spanning tree (highlighted)"
        elif mode == "mst":
            mst_edges = st.session_state.get("page3_mst_edges") or []
            highlight_edges = [(u, v) for (u, v, _) in mst_edges]
            total_w = sum(d.get("weight", 0) for (_, _, d) in mst_edges)
            title = f"MST by Kruskal (total weight = {total_w})"

        pos = choose_layout(G)
        fig, ax = plt.subplots(figsize=(7.0, 3.5))

        fig.patch.set_facecolor("#121212")
        ax.set_facecolor("#121212")

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color="#008cff",
            edgecolors="white",
            linewidths=1.5,
            node_size=BASE_NODE_SIZE,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color="#888888",
            width=1.8,
        )
        if highlight_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=highlight_edges,
                ax=ax,
                edge_color="#ff5252",
                width=3.0,
            )

        edge_labels = {(u, v): d.get("weight", "") for u, v, d in G.edges(data=True)}
        nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_size=10)
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_color="#ffd54f",
            font_size=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="#303030", ec="none"),
        )

        ax.set_title(title, color="white", pad=8)
        ax.axis("off")
        st.pyplot(fig)

# ----------------- Page 4: Planarity, Euler and Hamilton ----------------- #
def page_planarity_euler_hamilton():
    st.header("Page 4: Planarity, Eulerian & Hamiltonian")

    st.markdown(
        """
Here you can generate small simple graphs check for **planarity**, **Euler trails/circuits**, and **Hamiltonian cycles**.

#### To use:
1. Enter an unweighted edge list.
2. Build the graph.
3. Use the tools to test planarity and Euler/Hamilton properties and view example paths (when they exist).
---
"""
    )

    if "page4_graph" not in st.session_state:
        st.session_state["page4_graph"] = None
    if "page4_active_view" not in st.session_state:
        st.session_state["page4_active_view"] = "plain"  # "plain", "euler", "ham"
    if "page4_euler_path" not in st.session_state:
        st.session_state["page4_euler_path"] = None
    if "page4_ham_cycle" not in st.session_state:
        st.session_state["page4_ham_cycle"] = None

    # ---------- Row 1: input + tools ----------
    left_controls, right_tools = st.columns([1.1, 1.3])

    with left_controls:
        st.subheader("Graph input")

        st.caption("Simple graph, unweighted edges (one per line, format: `u v`):")
        example_text = "a b\na c\nb c\nb d\nc e\nd e"
        edge_text = st.text_area(
            "Edge list",
            value=example_text,
            height=140,
            key="page4_edge_text",
        )

        build_clicked = st.button(
            "Build / Update graph",
            use_container_width=True,
            key="page4_build_btn",
        )

        if build_clicked:
            G, err = parse_edge_list(
                edge_text,
                directed=False,
                weighted=False,
                max_nodes=MAX_NODES,
            )
            if err:
                st.error(err)
                st.session_state["page4_graph"] = None
                st.session_state["page4_active_view"] = "plain"
                st.session_state["page4_euler_path"] = None
                st.session_state["page4_ham_cycle"] = None
            else:
                st.session_state["page4_graph"] = G
                st.session_state["page4_active_view"] = "plain"
                st.session_state["page4_euler_path"] = None
                st.session_state["page4_ham_cycle"] = None

    with right_tools:
        st.subheader("Planarity & trails/cycles")

        G = st.session_state.get("page4_graph")
        if not G:
            st.info("Build a graph on the left to enable the tools.")
        elif G.number_of_nodes() == 0:
            st.info("Graph has no nodes.")
        else:
            # planarity test
            is_planar, embedding = nx.check_planarity(G, False)
            st.write(f"**Planar graph?** {'Yes' if is_planar else 'No'}")

            # Eulerian properties
            if nx.is_eulerian(G):
                euler_text = "Graph is Eulerian: has an `Euler circuit`."
            elif nx.has_eulerian_path(G):
                euler_text = "Graph has an `Euler trail` but **not** a `circuit`."
            else:
                euler_text = "Graph has **no** `Euler trail` or `circuit`."
            st.write(f"**Euler property:** {euler_text}")

            # Hamiltonian cycle (simple backtracking search, small graphs only)
            def find_hamiltonian_cycle(H: nx.Graph):
                n = H.number_of_nodes()
                if n == 0:
                    return None
                nodes = list(H.nodes())
                start = nodes[0]
                path = [start]
                used = {start}

                def backtrack(v):
                    if len(path) == n:
                        if start in H.neighbors(v):
                            return path + [start]
                        return None
                    for w in H.neighbors(v):
                        if w in used:
                            continue
                        used.add(w)
                        path.append(w)
                        res = backtrack(w)
                        if res is not None:
                            return res
                        path.pop()
                        used.remove(w)
                    return None

                return backtrack(start)

            ham_cycle = find_hamiltonian_cycle(G)
            if ham_cycle:
                st.write("**Hamiltonian cycle:** Yes (one example is shown when you pick the Hamiltonian view).")
            else:
                st.write("**Hamiltonian cycle:** No Hamiltonian cycle found (for this small search).")

            st.markdown("**Highlight view:**")
            col_plain, col_euler, col_ham = st.columns(3)
            with col_plain:
                plain_btn = st.button("Plain graph", use_container_width=True, key="page4_view_plain")
            with col_euler:
                euler_btn = st.button("Euler trail / circuit", use_container_width=True, key="page4_view_euler")
            with col_ham:
                ham_btn = st.button("Hamiltonian cycle", use_container_width=True, key="page4_view_ham")

            if plain_btn:
                st.session_state["page4_active_view"] = "plain"

            if euler_btn:
                try:
                    if nx.is_eulerian(G):
                        trail = list(nx.eulerian_circuit(G))
                    elif nx.has_eulerian_path(G):
                        trail = list(nx.eulerian_path(G))
                    else:
                        trail = None
                    if trail:
                        # convert edge sequence to list of edges
                        epath = [(u, v) for u, v in trail]
                        st.session_state["page4_euler_path"] = epath
                        st.session_state["page4_active_view"] = "euler"
                    else:
                        st.warning("No Euler trail or circuit exists for this graph.")
                        st.session_state["page4_euler_path"] = None
                except Exception as e:
                    st.error(f"Failed to compute Euler trail/circuit: {e}")
                    st.session_state["page4_euler_path"] = None

            if ham_btn:
                if ham_cycle:
                    # store as undirected edges along the cycle
                    edges = []
                    for i in range(len(ham_cycle) - 1):
                        u, v = ham_cycle[i], ham_cycle[i + 1]
                        edges.append((u, v))
                    st.session_state["page4_ham_cycle"] = edges
                    st.session_state["page4_active_view"] = "ham"
                else:
                    st.warning("No Hamiltonian cycle found to highlight.")
                    st.session_state["page4_ham_cycle"] = None

    st.markdown("---")

    # ---------- Row 2: centered graph view ----------
    G = st.session_state.get("page4_graph")
    if not G:
        st.info("Build a graph above to see the visualization.")
        return
    if G.number_of_nodes() == 0:
        st.info("Graph has no nodes.")
        return

    left_pad, center_col, right_pad = st.columns([1, 2, 1])
    with center_col:
        st.markdown("### <div style='text-align:center'>Graph view</div>", unsafe_allow_html=True)

        mode = st.session_state.get("page4_active_view", "plain")
        highlight_edges = []
        title = "Simple graph"

        if mode == "euler":
            highlight_edges = st.session_state.get("page4_euler_path") or []
            title = "Euler trail / circuit (highlighted)"
        elif mode == "ham":
            highlight_edges = st.session_state.get("page4_ham_cycle") or []
            title = "Hamiltonian cycle (highlighted)"

        pos = choose_layout(G)
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        fig.patch.set_facecolor("#121212")
        ax.set_facecolor("#121212")

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color="#008cff",
            edgecolors="white",
            linewidths=1.5,
            node_size=BASE_NODE_SIZE,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color="#888888",
            width=1.8,
        )
        if highlight_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=highlight_edges,
                ax=ax,
                edge_color="#ff5252",
                width=3.0,
            )

        nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_size=10)

        ax.set_title(title, color="white", pad=8)
        ax.axis("off")
        st.pyplot(fig)



# ----------------- Main ----------------- #

def main():
    st.sidebar.title("Discrete Structures Project")
    page = st.sidebar.radio(
        "Go to page:",
        [
            "1. Simple Graphs & Traversals",
            "2. Rooted & Binary Trees",
            "3. Spanning Trees & MSTs",
            "4. Planarity, Euler & Hamilton",
        ],
    )

    if page.startswith("1"):
        page_simple_graphs()
    elif page.startswith("2"):
        page_trees()
    elif page.startswith("3"):
        page_spanning_trees()
    else:
        page_planarity_euler_hamilton()


if __name__ == "__main__":
    main()
