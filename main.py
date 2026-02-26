import igraph as ig
import random
import math
import copy

# ============================================================
# Redistricting feasibility by Simulated Annealing (igraph)
# Author: Kusal Thapa
#
# Goal:
#   Partition a graph into k connected districts such that each
#   district population lies within [LOW, HIGH].
#
# Representation:
#   - Each node label is the population value itself (integer).
#   - Graph edges represent adjacency (must remain contiguous).
#
# Method :
#   1) Build an initial connected k-partition using region-growing.
#   2) Improve using Simulated Annealing:
#        - MOVE: move a boundary node from district A to B (preserve connectivity)
#        - SWAP: swap boundary nodes across two districts (preserve connectivity)
#   3) Multiple restarts; keep best (minimum penalty).
#
# Energy / Penalty:
#   penalty(pop) = distance outside [LOW, HIGH]
#   energy = sum penalty over all districts (energy = 0 means feasible)
#- Connectivity checks are done via BFS restricted to a district set (faster than inducing subgraphs
#    repeatedly), except when computing articulation points (igraph helper).
# ============================================================

random.seed(42)


def parse_pop(s: str) -> int:
    """Convert strings like '162,233' into int 162233."""
    return int(s.replace(",", "").strip())


def save_solution_to_file(problem_name, best_solution, energy_value, filename=None):
    """
    Save the final solution (district composition, populations, energy)
    to a clean text file for reporting/email.
    """

    if filename is None:
        # Create a clean filename automatically
        clean_name = problem_name.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"{clean_name}_solution.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Problem: {problem_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Districts: {len(best_solution)}\n")
        f.write(f"Energy (Population Violation): {energy_value}\n\n")

        for i, dset in enumerate(best_solution, 1):
            pop = sum(dset)
            f.write(f"District {i}\n")
            f.write(f"Population: {pop}\n")
            f.write(f"Number of Nodes: {len(dset)}\n")
            f.write("Nodes (Population Labels):\n")
            f.write(", ".join(str(x) for x in sorted(dset)))
            f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")

    print(f"\nSolution saved to file: {filename}")

# ============================================================
# SOLVER CORE 
# ============================================================

def run_one_problem(problem_name, nodes, edges, LOW, HIGH, k, restarts=60):
    """
    Solve one instance of the redistricting feasibility problem.

    Parameters
    ----------
    problem_name : str
        Name displayed in output.
    nodes : list[int]
        Node labels. In this dataset, labels are population values (integers).
    edges : list[tuple[int,int]]
        Undirected adjacency list using the same labels as in 'nodes'.
    LOW, HIGH : int
        Allowed population interval for each district.
    k : int
        Number of districts.
    restarts : int
        Number of random restarts (each restart generates a new initial partition).
    """

    # ----------------- Build igraph graph -----------------
    # label_to_idx allows fast mapping from population-label -> igraph vertex index
    label_to_idx = {lab: i for i, lab in enumerate(nodes)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    # Sanity check: all edge endpoints must exist in nodes
    missing = [(u, v) for (u, v) in edges if u not in label_to_idx or v not in label_to_idx]
    if missing:
        raise ValueError(
            f"[{problem_name}] Some edges reference nodes not in node list. "
            f"Example: {missing[:10]} (total missing={len(missing)})"
        )

    # Convert label-edges into index-edges for igraph
    edges_idx = [(label_to_idx[u], label_to_idx[v]) for (u, v) in edges]
    G = ig.Graph(n=len(nodes), edges=edges_idx, directed=False)

    # Store labels on vertices for induced subgraphs / articulation checks
    G.vs["label"] = nodes
    G.vs["name"] = [str(x) for x in nodes]

    # ----------------- Basic summary -----------------
    print("\n" + "=" * 90)
    print(f"Problem: {problem_name}")
    print("=" * 90)
    print("Nodes:", len(nodes), "Edges:", len(edges))
    print("Graph connected?:", G.is_connected())

    T = sum(nodes)
    print("Total population T =", T)
    print("Feasibility necessary condition (k*LOW <= T <= k*HIGH):", k * LOW <= T <= k * HIGH)
    print("Target per district approx:", T / k)

    # ----------------- Connectivity helper -----------------
    def is_connected_labels(graph: ig.Graph, labels_set: set[int]) -> bool:
        """
        Check whether the subgraph induced by 'labels_set' is connected.
        """
        if len(labels_set) <= 1:
            return True
        idxs = [label_to_idx[x] for x in labels_set]
        sub = graph.induced_subgraph(idxs)
        return sub.is_connected()

    # ----------------- Initial connected partition (region growing) -----------------
    def initial_connected_partition(graph: ig.Graph, k: int, max_restarts: int = 400):
        """
        Build an initial k-partition where each district is connected.
        Strategy:
          - Pick k random seed nodes
          - Grow each district by repeatedly adding neighboring unassigned nodes
          - Restart if we get stuck (cannot expand)
        """
        labels = graph.vs["label"]
        for _ in range(max_restarts):
            seeds = random.sample(labels, k)
            districts = [set([s]) for s in seeds]
            unassigned = set(labels) - set(seeds)

            stuck = 0
            while unassigned:
                progress = False

                # Expand each district one step (round-robin)
                for d in range(k):
                    frontier = set()
                    for v_lab in districts[d]:
                        v_idx = label_to_idx[v_lab]
                        for nbr_idx in graph.neighbors(v_idx):
                            frontier.add(idx_to_label[nbr_idx])

                    frontier &= unassigned

                    if frontier:
                        u = random.choice(list(frontier))
                        districts[d].add(u)
                        unassigned.remove(u)
                        progress = True
                        if not unassigned:
                            break

                # If no district could grow, we are stuck → restart
                if not progress:
                    stuck += 1
                    if stuck > 20:
                        break

            # Accept only if all nodes assigned and each district is connected
            if (not unassigned) and all(is_connected_labels(graph, dset) for dset in districts):
                return districts

        return None

    # ----------------- Energy / penalty -----------------
    def penalty(p, low=LOW, high=HIGH):
        """
        Distance outside the allowed interval [low, high].
        0 means the district is within bounds.
        """
        if p < low:
            return low - p
        if p > high:
            return p - high
        return 0

    def energy_k(districts):
        """
        Total energy (sum of penalties).
        Energy == 0  → fully feasible population bounds.
        """
        return sum(penalty(sum(d)) for d in districts)

    # ----------------- Local move generation -----------------
    def boundary_nodes_between(graph: ig.Graph, Da: set[int], Db: set[int]):
        """
        Nodes in district Da that have at least one neighbor in district Db.
        Only these are candidates for moving from Da to Db while preserving adjacency structure.
        """
        Db_set = set(Db)
        out = []
        for v_lab in Da:
            v_idx = label_to_idx[v_lab]
            if any(idx_to_label[nbr] in Db_set for nbr in graph.neighbors(v_idx)):
                out.append(v_lab)
        return out

    def safe_removable_nodes(graph: ig.Graph, Dset: set[int]):
        """
        Prefer nodes that are not articulation points of the district subgraph.
        Removing an articulation point could disconnect the district.

        We compute articulation points in the induced subgraph and map back to labels.
        """
        if len(Dset) <= 2:
            return list(Dset)

        idxs = [label_to_idx[x] for x in Dset]
        sub = graph.induced_subgraph(idxs)
        arts_sub = set(sub.articulation_points())

        # Convert articulation indices in the induced subgraph back to labels
        arts_labels = {int(sub.vs[i]["label"]) for i in arts_sub}

        return [v for v in Dset if v not in arts_labels]

    def try_move_between(graph: ig.Graph, districts, a, b, max_tries=120):
        """
        Attempt to move one boundary node from district a -> b, preserving connectivity.
        Returns:
          - (new_districts, moved_node) if successful
          - None if no valid move found
        """
        Da = set(districts[a])
        Db = set(districts[b])

        if len(Da) <= 1:
            return None

        cand = boundary_nodes_between(graph, Da, Db)
        if not cand:
            return None

        # Prioritize nodes that are "safe" to remove (not articulation points)
        safe_a = list(set(cand) & set(safe_removable_nodes(graph, Da)))
        cand = safe_a + [x for x in cand if x not in safe_a]

        random.shuffle(cand)

        for v in cand[:max_tries]:
            new_Da = set(Da)
            new_Db = set(Db)
            new_Da.remove(v)
            new_Db.add(v)

            # Contiguity constraints for both districts
            if is_connected_labels(graph, new_Da) and is_connected_labels(graph, new_Db):
                new_districts = [set(d) for d in districts]
                new_districts[a] = new_Da
                new_districts[b] = new_Db
                return new_districts, v

        return None

    def try_swap_between(graph: ig.Graph, districts, a, b, max_tries=80):
        """
        Attempt to swap boundary nodes between districts a and b, preserving connectivity.
        Returns:
          - (new_districts, (v_from_a, u_from_b)) if successful
          - None otherwise
        """
        Da = set(districts[a])
        Db = set(districts[b])

        boundary_a = boundary_nodes_between(graph, Da, Db)
        boundary_b = boundary_nodes_between(graph, Db, Da)
        if not boundary_a or not boundary_b:
            return None

        # Only consider "safe" boundary nodes on each side
        safe_a = list(set(boundary_a) & set(safe_removable_nodes(graph, Da)))
        safe_b = list(set(boundary_b) & set(safe_removable_nodes(graph, Db)))
        if not safe_a or not safe_b:
            return None

        random.shuffle(safe_a)
        random.shuffle(safe_b)

        for v in safe_a[:max_tries]:
            for u in safe_b[:max_tries]:
                new_Da = (Da - {v}) | {u}
                new_Db = (Db - {u}) | {v}

                if is_connected_labels(graph, new_Da) and is_connected_labels(graph, new_Db):
                    new_districts = [set(d) for d in districts]
                    new_districts[a] = new_Da
                    new_districts[b] = new_Db
                    return new_districts, (v, u)

        return None

    # ----------------- Simulated annealing -----------------
    def simulated_annealing(graph: ig.Graph, districts,
                            T0=3e6, alpha=0.9997, max_iter=500000):
        """
        Simulated annealing on the space of connected partitions.
        We accept:
          - any improving move (dE <= 0), or
          - worsening moves with probability exp(-dE / T)
        """
        current = [set(d) for d in districts]
        current_E = energy_k(current)

        best = copy.deepcopy(current)
        best_E = current_E

        T = T0

        for it in range(max_iter):
            if best_E == 0:
                print("Feasible (energy 0) found at iteration", it)
                return best

            # pick two distinct districts at random
            a, b = random.sample(range(len(current)), 2)

            # choose move type: MOVE (more often) vs SWAP
            if random.random() < 0.65:
                trial = try_move_between(graph, current, a, b)
            else:
                trial = try_swap_between(graph, current, a, b)

            if trial is None:
                T *= alpha
                continue

            new_districts, _ = trial
            new_E = energy_k(new_districts)
            dE = new_E - current_E

            # SA acceptance rule
            if dE <= 0 or random.random() < math.exp(-dE / max(T, 1e-12)):
                current = new_districts
                current_E = new_E

                # record best so far
                if current_E < best_E:
                    best = copy.deepcopy(current)
                    best_E = current_E

            T *= alpha

        print("Finished SA. Best energy:", best_E)
        return best

    def solve_with_restarts(graph: ig.Graph, k, restarts=restarts):
        """
        Multiple restarts:
          - generate a fresh initial connected partition
          - run SA
          - keep the best (lowest energy) result
        """
        best_sol = None
        best_E = float("inf")

        for r in range(restarts):
            districts0 = initial_connected_partition(graph, k)
            if districts0 is None:
                print(f"Restart {r+1}/{restarts}: could not build initial partition")
                continue

            sol = simulated_annealing(graph, districts0)
            E = energy_k(sol)

            if E < best_E:
                best_E = E
                best_sol = sol

            pops = [sum(d) for d in sol]
            print(f"Restart {r+1}/{restarts} | run energy={E} | best={best_E} | pops={pops}")

            if best_E == 0:
                break

        return best_sol

    # ----------------- Run + print final -----------------
    best_solution = solve_with_restarts(G, k, restarts=restarts)

    print("\nBEST pops:", [sum(d) for d in best_solution])
    print("BEST energy:", energy_k(best_solution))
    print("All connected?", all(is_connected_labels(G, d) for d in best_solution))
    # Save best solution to file
    final_energy = energy_k(best_solution)
    save_solution_to_file(problem_name, best_solution, final_energy)

    for i, dset in enumerate(best_solution, 1):
        print(f"\nDistrict {i} pop={sum(dset)} size={len(dset)}")
        print(sorted(dset))


# ============================================================
# DATA: South Carolina, Oregon, Maine (unchanged)
# ============================================================

# 1) SOUTH CAROLINA
sc_nodes_str = [
    "162,233", "24,777", "21,090", "38,892", "136,555", "350,209", "177,843",
    "60,158", "34,423", "33,062", "269,291", "32,062", "28,933", "136,885",
    "34,971", "92,501", "15,987", "10,419", "22,621", "160,099", "26,985",
    "10,233", "19,875", "262,391", "15,175", "107,456", "19,220", "68,681",
    "46,734", "76,652", "61,697", "384,504", "23,956", "37,508", "69,661",
    "25,417", "66,537", "28,961", "33,140", "226,073", "55,342", "284,307",
    "451,225", "187,126", "119,224", "74,273"
]
sc_nodes = [parse_pop(x) for x in sc_nodes_str]

sc_edges_str = [
    ("162,233", "24,777"), ("162,233", "38,892"), ("162,233", "21,090"), ("24,777", "21,090"),
    ("21,090", "38,892"), ("10,419", "21,090"), ("10,419", "15,987"),
    ("10,419", "22,621"), ("15,987", "38,892"), ("15,987", "92,501"), ("15,987", "22,621"),

    ("38,892", "136,555"), ("136,555", "350,209"), ("38,892", "350,209"),
    ("136,555", "92,501"), ("38,892", "92,501"),
    ("350,209", "177,843"), ("350,209", "60,158"), ("60,158", "34,423"),
    ("60,158", "269,291"), ("60,158", "33,062"), ("60,158", "177,843"),
    ("269,291", "33,062"), ("269,291", "32,062"),

    ("92,501", "177,843"), ("92,501", "34,971"), ("92,501", "22,621"),
    ("92,501", "160,099"), ("92,501", "15,175"), ("177,843", "34,423"),
    ("177,843", "136,555"), ("177,843", "34,971"), ("92,501", "262,391"),

    ("33,062", "136,885"), ("33,062", "32,062"), ("33,062", "34,423"),
    ("136,885", "32,062"), ("136,885", "34,423"), ("32,062", "28,933"),
    ("28,933", "68,681"), ("28,933", "136,885"), ("68,681", "46,734"),
    ("68,681", "136,885"), ("46,734", "76,652"), ("46,734", "28,933"),
    ("34,971", "136,885"), ("34,971", "15,175"), ("34,971", "34,423"),

    ("15,175", "107,456"), ("107,456", "136,885"), ("107,456", "19,220"),
    ("107,456", "34,971"), ("107,456", "61,697"),
    ("19,220", "68,681"), ("19,220", "136,885"), ("19,220", "61,697"),
    ("61,697", "46,734"), ("61,697", "68,681"),

    ("160,099", "262,391"), ("160,099", "22,621"), ("160,099", "19,875"),
    ("262,391", "15,175"), ("384,504", "15,175"),
    ("262,391", "384,504"), ("384,504", "107,456"), ("384,504", "61,697"),
    ("384,504", "23,956"), ("23,956", "61,697"), ("23,956", "76,652"),
    ("23,956", "33,140"),

    ("26,985", "160,099"), ("26,985", "19,875"), ("26,985", "10,233"), ("26,985", "69,661"),
    ("19,875", "262,391"), ("19,875", "37,508"), ("37,508", "384,504"),
    ("37,508", "23,956"), ("37,508", "262,391"), ("37,508", "28,961"),
    ("28,961", "23,956"), ("28,961", "284,307"), ("28,961", "226,073"), ("28,961", "55,342"),
    ("28,961", "33,140"), ("33,140", "76,652"), ("33,140", "226,073"),
    ("76,652", "226,073"), ("61,697", "76,652"),

    ("10,233", "69,661"), ("10,233", "25,417"), ("69,661", "19,875"),
    ("69,661", "37,508"), ("69,661", "66,537"), ("69,661", "25,417"),
    ("25,417", "66,537"), ("25,417", "187,126"), ("66,537", "37,508"),
    ("66,537", "28,961"), ("66,537", "284,307"), ("284,307", "55,342"),
    ("55,342", "226,073"), ("284,307", "451,225"), ("451,225", "187,126"),
    ("187,126", "74,273"), ("187,126", "119,224"), ("119,224", "74,273"),
    ("451,225", "119,224"), ("66,537", "451,225")
]
sc_edges = [(parse_pop(a), parse_pop(b)) for (a, b) in sc_edges_str]

# 2) OREGON
or_nodes = [
    82713, 22364, 63043, 107667, 203206, 66380, 7895, 351715, 157733,
    7422, 31313, 16134, 7008, 25748, 75889, 7445, 20978, 46034, 85579,
    116672, 21720, 1441, 11173, 1871, 1765, 25213, 315335, 75403,
    25250, 37039, 529710, 99193, 375992, 49351, 735334, 22346
]

or_edges = [
    (82713, 22364), (82713, 107667), (82713, 203206),
    (22364, 63043), (22364, 107667),
    (63043, 107667),
    (203206, 107667), (203206, 66380),
    (66380, 107667),(66380, 157733), (66380, 351715), (66380, 7895),
    (7895, 157733), (7895, 7422),
    (351715, 107667), (351715, 46034), (351715, 85579), (351715, 157733),
    (157733, 116672), (157733, 20978), (157733, 7422),
    (7422, 31313), (7422, 20978), (7422, 7445),
    (31313, 16134), (31313, 7445),
    (16134, 7008), (16134, 25748),
    (7008, 25748), (7008, 75889),
    (25748, 7445), (25748, 75889),
    (75889, 7445), (75889, 11173),
    (7445, 20978), (7445, 1441), (7445, 16134), (7445, 11173),
    (20978, 1441), (20978, 21720), (46034, 25250), (46034, 75403), (46034, 85579),
    (85579, 116672), (85579, 75403),
    (116672, 21720), (116672, 315335), (116672, 351715),
    (21720, 1441), (21720, 25213), (21720, 157733), (21720, 315335),
    (1441, 11173), (1441, 1871), (1441, 25213),
    (11173, 1871),
    (1871, 1765), (1871, 25213),
    (1765, 25213),
    (25213, 315335), (25213, 375992), (25213, 22346),
    (315335, 75403), (315335, 99193), (315335, 375992),
    (75403, 25250), (75403, 99193), (75403, 116672),
    (25250, 37039), (25250, 99193), (25250, 529710), (37039, 49351),
    (529710, 99193), (529710, 375992), (529710, 735334), (529710, 49351),
    (99193, 375992),
    (375992, 735334), (375992, 22346),
    (49351, 735334),
    (735334, 22346)
]

# 3) MAINE
me_nodes = [
    32856, 71870, 54418, 153923, 17535, 39736, 38786, 52228,
    34457, 122151, 30768, 57833, 35293, 107702, 281674, 197131
]

me_edges = [
    (32856, 54418), (32856, 153923), (32856, 71870),
    (54418, 153923), (54418, 38786), (54418, 39736),
    (153923, 38786), (153923, 17535), (153923, 52228), (153923, 71870),
    (71870, 17535), (71870, 52228),
    (17535, 52228),
    (39736, 38786), (39736, 34457),
    (38786, 34457), (38786, 122151), (38786, 52228),
    (34457, 122151), (34457, 35293),
    (122151, 52228), (122151, 30768), (122151, 107702), (122151, 35293),
    (52228, 30768),
    (30768, 57833), (30768, 107702),
    (35293, 107702), (35293, 281674),
    (107702, 57833), (107702, 281674),
    (57833, 281674), (57833, 197131),
    (281674, 197131),
]


# ============================================================
# RUNNER (clean toggle; does not change algorithm)
# ============================================================

if __name__ == "__main__":
    # South Carolina
    run_one_problem(
        problem_name="South Carolina (k=7)",
        nodes=sc_nodes,
        edges=sc_edges,
        LOW=657_463, HIGH=664_070,
        k=7,
        restarts=8
    )


    run_one_problem(
         problem_name="Oregon (k=5)",
         nodes=or_nodes,
         edges=or_edges,
         LOW=762_384, HIGH=770_045,
         k=5,
         restarts=20
     )
    
    run_one_problem(
         problem_name="Maine (k=2)",
         nodes=me_nodes,
         edges=me_edges,
         LOW=660_860, HIGH=667_501,
         k=2,
         restarts=40
     )
    
