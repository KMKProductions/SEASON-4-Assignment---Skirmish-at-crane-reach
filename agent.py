"""Skirmish at Crane Reach agent built from reusable behavior blocks.

Read ``environment.md`` beside this file for the rules and helpers. Every unit runs its own
``Agent`` instance, so the army never shares memory. Each unit instead derives the same army plan
from facts every ally agrees on: the roster, the battlefield, and the current round.

The army plays for the capture zone, which sits on the river seam at the field center. One
footman runs straight to the zone and holds it. The other footmen and the archers regroup just short
of the zone, then move up together. There the footmen fill every zone tile and never leave it,
with one spare footman in reserve to fill a gap, while the archers stand back, climbing a nearby
hill when an enemy is in range. The cavalry splits into two wings that wait at the mouths of the
outer river passages, then ride through them to flank the enemy beside the zone, hunting archers
first. A rider that sees an enemy stops waiting and fights at once.

Every move toward a goal is routed over the whole field, so units walk around the river through
a passage instead of stalling on its bank.

An ``Agent`` is only a playbook: an ordered list of behavior blocks per unit type. The first block
that proposes an order wins. Blocks are small, stateless, and reusable across unit types.
"""

from __future__ import annotations

import heapq

from sandbox.crane import action, me, paths, roster, tile, units, visible
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation

_BAD_TERRAIN = {"marsh", "waste"}
_WATER = {"water"}
_IMPASSABLE = {"water", "void"}

# Movement point cost to enter a tile, from the rules table in environment.md.
_TERRAIN_COST = {"grass": 1, "hill": 2}
_FEATURE_COST = {"none": 0, "forest": 1, "marsh": 2, "waste": 0}
# Extra routing cost for ground that hurts on entry, so routes avoid it when they can.
_ROUTE_PENALTY = {"waste": 2}

ENGAGE_RADIUS = 6  # an enemy this close ends the march and starts the fight
_UNREACHABLE = 10**9

Hex = tuple[int, int]

# Offsets that fill a hex outward from its center, ring by ring. The first seven are a center and
# its six neighbors: the shape of a capture zone.
_CLUSTER: list[Hex] = sorted(
    ((dq, dr) for dq in range(-3, 4) for dr in range(-3, 4) if (abs(dq) + abs(dr) + abs(dq + dr)) // 2 <= 3),
    key=lambda offset: ((abs(offset[0]) + abs(offset[1]) + abs(offset[0] + offset[1])) // 2, offset),
)


def _hex(position: AxialPosition) -> Hex:
    return position["q"], position["r"]


def _hex_neighbors(position: Hex) -> list[Hex]:
    return [(position[0] + dq, position[1] + dr) for dq, dr in tile.DIRECTIONS.values()]


def _hex_distance(first: Hex, second: Hex) -> int:
    dq = first[0] - second[0]
    dr = first[1] - second[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _cube_round(fq: float, fr: float) -> Hex:
    fs = -fq - fr
    q, r, s = round(fq), round(fr), round(fs)
    dq, dr, ds = abs(q - fq), abs(r - fr), abs(s - fs)
    if dq > dr and dq > ds:
        q = -r - s
    elif dr > ds:
        r = -q - s
    return q, r


def _hex_line(start: Hex, end: Hex) -> list[Hex]:
    """Return every tile on the straight hex line from start to end, both included."""
    steps = _hex_distance(start, end)
    if steps == 0:
        return [start]
    # The tiny nudge keeps rounding consistent when the line runs exactly along a tile edge.
    return [
        _cube_round(
            start[0] + (end[0] - start[0]) * i / steps + 1e-6,
            start[1] + (end[1] - start[1]) * i / steps + 1e-6,
        )
        for i in range(steps + 1)
    ]


# --------------------------------------------------------------------------------------------
# Whole-field routing
# --------------------------------------------------------------------------------------------


class TerrainMap:
    """Travel costs over the whole field for one match, so routes go around the river.

    The action mask only covers paths of up to four steps, which cannot see a passage further
    down the bank. ``field`` runs one cheapest-route search from a goal over every passable tile,
    and a unit then picks the legal path whose landing has the cheapest remaining route.
    """

    CACHE_LIMIT = 256  # goals whose cost fields are kept for reuse within the match
    OCCUPIED_PENALTY = 5  # extra cost for a tile a unit stands on: it may move, but likely not

    def __init__(self, observation: SkirmishObservation) -> None:
        battlefield = observation["observation"]["battlefield"]
        view = {"observation": {"battlefield": battlefield}}
        side = battlefield["side"]
        self.entry_cost: dict[Hex, int] = {}
        self._route_cost: dict[Hex, int] = {}
        self.hills: frozenset[Hex] = frozenset(
            (q, r)
            for q in range(side)
            for r in range(side)
            if tile.terrain_at(view, {"q": q, "r": r})["terrain"] == "hill"
        )
        for q in range(side):
            for r in range(side):
                ground = tile.terrain_at(view, {"q": q, "r": r})
                if ground["terrain"] in _IMPASSABLE:
                    continue
                cost = _TERRAIN_COST.get(ground["terrain"], 1) + _FEATURE_COST.get(ground["feature"], 0)
                self.entry_cost[(q, r)] = cost
                self._route_cost[(q, r)] = cost + _ROUTE_PENALTY.get(ground["feature"], 0)
        self._fields: dict[Hex, dict[Hex, int]] = {}

    def passable(self, position: Hex) -> bool:
        return position in self.entry_cost

    def field(self, goal: Hex, occupied: frozenset[Hex] = frozenset()) -> dict[Hex, int]:
        """Return the cheapest route cost from every reachable tile to ``goal``.

        Tiles in ``occupied`` cost extra, so routes flow around units standing in a narrow
        passage. Those fields depend on the moment, so only unoccupied ones are cached.
        """
        if not occupied:
            cached = self._fields.get(goal)
            if cached is not None:
                return cached
            if len(self._fields) >= self.CACHE_LIMIT:
                self._fields.clear()

        # An impassable goal, such as water, is approached from its passable neighbors.
        seeds = [goal] if self.passable(goal) else [n for n in _hex_neighbors(goal) if self.passable(n)]
        costs = {seed: 0 for seed in seeds}
        frontier = [(0, seed) for seed in seeds]
        while frontier:
            cost, position = heapq.heappop(frontier)
            if cost > costs[position]:
                continue
            # Searching backward from the goal: a neighbor reaches here by entering this tile.
            through = cost + self._route_cost[position]
            if position in occupied:
                through += self.OCCUPIED_PENALTY
            for neighbor in _hex_neighbors(position):
                if self.passable(neighbor) and through < costs.get(neighbor, _UNREACHABLE):
                    costs[neighbor] = through
                    heapq.heappush(frontier, (through, neighbor))

        if not occupied:
            self._fields[goal] = costs
        return costs

    def path_cost(self, start: Hex, path_id: int) -> int:
        """Return the movement points a path spends, entering each tile along it."""
        q, r = start
        total = 0
        for digit in paths.decode(path_id):
            dq, dr = tile.DIRECTIONS[digit]
            q, r = q + dq, r + dr
            total += self.entry_cost.get((q, r), _UNREACHABLE)
        return total


# --------------------------------------------------------------------------------------------
# Movement primitives
# --------------------------------------------------------------------------------------------


class Navigator:
    """Path choices for one activation. Every path it returns comes from the legal mask."""

    def __init__(self, observation: SkirmishObservation, terrain: TerrainMap) -> None:
        self.observation = observation
        self.terrain = terrain
        self.here = me.position(observation)
        self._occupied = frozenset(_hex(unit["position"]) for unit in observation["observation"]["visible_units"])

    def field(self, goal: Hex) -> dict[Hex, int]:
        """Return route costs to goal that steer around the units this unit can see."""
        return self.terrain.field(goal, self._occupied - {goal})

    def path_has_bad_terrain(self, path_id: int) -> bool:
        """True when the path enters water, marsh, or waste terrain."""
        position = self.here
        for digit in paths.decode(path_id):
            dq, dr = tile.DIRECTIONS[digit]
            position = {"q": position["q"] + dq, "r": position["r"] + dr}
            terrain = tile.terrain_at(self.observation, position)
            if terrain["terrain"] in _BAD_TERRAIN | _WATER or terrain["feature"] in _BAD_TERRAIN | _WATER:
                return True
        return False

    def best_path(self, key, max_cost: int | None = None) -> int:
        """Return the legal path whose landing tile has the lowest ``key``, or 0 to stay.

        Standing still wins ties, and shorter paths come first in the mask, so a unit never walks
        further than the gain needs. ``max_cost`` limits the movement points the path may spend.
        """
        here = _hex(self.here)
        best_path, best_key = 0, key(here)
        for path_id in action.legal_paths(self.observation):
            if path_id == 0:
                continue
            if max_cost is not None and self.terrain.path_cost(here, path_id) > max_cost:
                continue
            candidate = key(_hex(tile.at_path_end(self.here, path_id)))
            if candidate < best_key:
                best_path, best_key = path_id, candidate
        return best_path

    def step_toward(self, goal: AxialPosition, max_cost: int | None = None) -> int:
        """Return the legal path that gets furthest along the real route to goal, or 0.

        The route comes from the whole field, so a goal across the river leads through the
        nearest passage instead of to the bank opposite it.
        """
        target = _hex(goal)
        field = self.field(target)
        return self.best_path(
            lambda landing: (field.get(landing, _UNREACHABLE), _hex_distance(landing, target)), max_cost
        )

    def travel_cost(self, start: AxialPosition, goal: AxialPosition) -> int:
        """Return the movement points the cheapest route from start to goal spends."""
        return self.terrain.field(_hex(goal)).get(_hex(start), _UNREACHABLE)

    def path_away(self, threat: AxialPosition) -> int:
        """Return the legal path that maximizes distance from a threat, avoiding bad terrain."""
        best_path = 0
        best_distance = tile.distance(self.here, threat)

        for path_id in action.legal_paths(self.observation):
            if path_id == 0 or self.path_has_bad_terrain(path_id):
                continue
            landing_distance = tile.distance(tile.at_path_end(self.here, path_id), threat)
            if landing_distance > best_distance:
                best_path, best_distance = path_id, landing_distance

        return best_path

    def flank_goal(self, enemy: AxialPosition) -> AxialPosition:
        """Choose the nearer of the two side tiles next to an enemy."""
        enemy_sides = tile.neighbors(enemy)
        side_tiles = (enemy_sides[1], enemy_sides[3])
        return min(side_tiles, key=lambda side: tile.distance(self.here, side))


# --------------------------------------------------------------------------------------------
# Army plan: the zone runner, the infantry's regroup and march, and the cavalry wings
# --------------------------------------------------------------------------------------------


class ArmyPlan:
    """Gives every unit a role and a tile for each round, all built around the capture zone.

    Every unit builds this from shared match knowledge only, so all allies agree on it without
    messaging. The plan lives in red's frame, where the enemy lies toward +q, and is mirrored
    through the field center for blue.

    * The runner, the footman with the shortest route to the zone, heads straight for its center.
    * The other footmen and every archer form up at a rally point just short of the zone, the
      footmen packed into a zone-shaped hex and the archers in a line behind them, then take their
      posts together. The footmen take the zone's other
      tiles, with one spare footman in reserve behind it, and the archers take posts further back
      that stay within bow range of the zone.
    * The cavalry splits into a north and a south wing. Each waits at our mouth of the outer river
      passage on its side until the infantry reaches the zone, then rides through it and swings in
      beside the zone's far side to flank whoever contests it.
    """

    FILE_SPACING = 2  # rows between neighboring archers: 2 leaves one open tile between them
    RANK_SPACING = 2  # columns from the footman block's back edge to the archers: 2 leaves one open
    RALLY_SEARCH = 6  # how far a rally point may shift from its spawn centroid to fit the group
    REGROUP_SLACK = 2  # extra rounds allowed for forming up, for terrain and traffic
    MARCH_STOP = 3  # columns short of the zone center where the infantry forms up
    RESERVE_DEPTH = 2  # columns behind the zone center where the spare footman waits
    ARCHER_BAND = (3, 5)  # nearest and furthest an archer post may be from the zone center
    ARCHER_GAP = 2  # least distance between two archer posts
    STAGE_DEPTH = 2  # columns short of the river where a cavalry wing waits at its passage
    STRIKE_REACH = 3  # tiles beyond the zone's far corners where the wings swing in
    SCOUT_AFTER = 10  # rounds past arrival, without contact, before archers and cavalry scout

    def __init__(self, observation: SkirmishObservation, terrain: TerrainMap) -> None:
        self.terrain = terrain
        self.side = observation["observation"]["battlefield"]["side"]
        self.extent = (self.side - 1) // 2
        self.is_red = me.side(observation) == "red"
        # The march moves at the pace of the slowest line troops, spending their full points.
        self.pace = min(units.STATS[kind].movement_points for kind in ("footman", "archer"))

        zones = observation["observation"]["battlefield"]["zones"]
        if zones:
            self.zone_center = self._to_red(zones[0]["center"])
            zone_tiles = {self._to_red(spot) for spot in zones[0]["tiles"]}
        else:
            self.zone_center = (self.extent, self.extent)
            zone_tiles = {self.zone_center, *(n for n in _hex_neighbors(self.zone_center) if terrain.passable(n))}
        self.zone_tiles = frozenset(zone_tiles)
        self.zone_board = frozenset(_hex(self._to_board(*position)) for position in zone_tiles)

        roster_order = [(entry["type"], int(entry["unit_id"].rsplit("_", 1)[1])) for entry in roster.allies(observation)]
        spawns = dict(zip(roster_order, self._spawns(len(roster_order))))
        own_key = (me.unit_type(observation), int(me.unit_id(observation).rsplit("_", 1)[1]))
        if spawns.get(own_key) != self._to_red(me.position(observation)):
            spawns = None  # the spawn rule changed, so plan without knowing where allies start

        by_kind: dict[str, list[int]] = {}
        for kind, index in roster_order:
            by_kind.setdefault(kind, []).append(index)
        footmen = by_kind.get("footman", [])
        to_zone = terrain.field(self.zone_center)
        if spawns is not None and footmen:
            self.runner = min(footmen, key=lambda index: (to_zone.get(spawns[("footman", index)], _UNREACHABLE), index))
        else:
            self.runner = footmen[0] if footmen else None
        self.infantry = {
            "footman": [index for index in footmen if index != self.runner],
            "archer": by_kind.get("archer", []),
        }

        self._plan_infantry(spawns)
        self._plan_posts()
        self._plan_wings(by_kind.get("cavalry", []), spawns)

    # -- Lookups used by Context ----------------------------------------------------------------

    def is_runner(self, kind: str, index: int) -> bool:
        return kind == "footman" and index == self.runner

    def wing(self, index: int) -> int:
        return self._wings[index][0]

    def slot(
        self, kind: str, index: int, round_number: int, released: bool = False, advanced: bool = False
    ) -> AxialPosition:
        """Return where this unit belongs this round, in board coordinates.

        A ``released`` rider skips its wait at the staging tile and heads for its strike tile. An
        ``advanced`` footman or archer skips the rest of the regroup and heads for its post.
        """
        if kind == "cavalry":
            _, stage, strike = self._wings[index]
            waiting = round_number <= self.arrive_round and not released
            return self._to_board(*(stage if waiting else strike))
        if self.is_runner(kind, index):
            return self._to_board(*self.zone_center)
        slot_index = self.infantry[kind].index(index)
        if advanced:
            position = self._posts[(kind, slot_index)]
        elif round_number <= self.ready_round:
            position = self._formation(self.infantry, self.rally)[(kind, slot_index)]
        elif round_number <= self.arrive_round:
            position = self._formation(self.infantry, self._anchor(round_number))[(kind, slot_index)]
        else:
            position = self._posts[(kind, slot_index)]
        return self._to_board(*position)

    def back_tiles(self) -> list[Hex]:
        """The zone's tiles nearest our own side, in board coordinates."""
        nearest = min(self.depth(cell) for cell in self.zone_board)
        return sorted(cell for cell in self.zone_board if self.depth(cell) == nearest)

    def rally_tiles(self, kind: str) -> frozenset[Hex]:
        """The board tiles one unit type's slots cover at the rally."""
        slots = self._formation(self.infantry, self.rally)
        return frozenset(
            _hex(self._to_board(*position)) for (slot_kind, _), position in slots.items() if slot_kind == kind
        )

    def depth(self, position: Hex) -> int:
        """How far a board tile lies toward the enemy: its column in red's frame."""
        return self._to_red({"q": position[0], "r": position[1]})[0]

    def on_our_side(self, position: Hex) -> bool:
        """True for a tile on our half of the river or on the seam itself."""
        return self.depth(position) <= self.zone_center[0]

    def scouting(self, round_number: int, last_contact: int | None) -> bool:
        """True once the infantry has held the zone for a while and this unit has seen nobody."""
        if round_number <= self.arrive_round + self.SCOUT_AFTER:
            return False
        return last_contact is None or round_number - last_contact > self.SCOUT_AFTER

    # -- Planning ---------------------------------------------------------------------------------

    def _plan_infantry(self, spawns: dict | None) -> None:
        """Choose the infantry's rally point, when it has formed, and its march to the zone."""
        members = self.infantry
        own = [spawns[(kind, index)] for kind, indices in members.items() for index in indices] if spawns else []
        # Form up just short of the zone, so the infantry can take its posts as soon as it is done.
        stop = self._onto_field(self.zone_center[0] - self.MARCH_STOP, self.zone_center[1])
        self.rally = self._choose_rally(members, stop)

        if own:
            slowest = 0
            rally_slots = self._formation(members, self.rally)
            for kind, indices in members.items():
                speed = units.STATS[kind].movement_points
                for slot_index, index in enumerate(indices):
                    walk = _hex_distance(spawns[(kind, index)], rally_slots[(kind, slot_index)])
                    slowest = max(slowest, -(-walk // speed))
            self.ready_round = slowest + self.REGROUP_SLACK
        else:
            self.ready_round = 8

        self.route = self._route([self.rally, stop])
        self.schedule = self._march_schedule(self.route)
        self.arrive_round = self.ready_round + len(self.schedule) - 1

    def _plan_posts(self) -> None:
        """Give every infantry slot the post it takes once the march is over."""
        march_end = self._formation(self.infantry, self.route[-1])
        center_q, center_r = self.zone_center
        # The zone's far edge fills first, since that is where the enemy arrives.
        ring = sorted(
            self.zone_tiles - {self.zone_center},
            key=lambda position: (-position[0], abs(position[1] - center_r), position[1]),
        )
        reserve = self._settle(center_q - self.RESERVE_DEPTH, center_r, center_r, set(self.zone_tiles))
        archer_posts = self._archer_posts(set(self.zone_tiles) | {reserve}, len(self.infantry["archer"]))
        self._posts = {
            **self._assign("footman", [*ring, reserve], march_end),
            **self._assign("archer", archer_posts, march_end),
        }

    def _archer_posts(self, taken: set[Hex], count: int) -> list[Hex]:
        """Pick spaced-out tiles behind the zone and within bow range of it, hills first."""
        nearest, furthest = self.ARCHER_BAND
        candidates = sorted(
            (
                position
                for position in self.terrain.entry_cost
                if position[0] < self.zone_center[0] - 1
                and nearest <= _hex_distance(position, self.zone_center) <= furthest
                and position not in taken
            ),
            key=lambda position: (
                position not in self.terrain.hills,
                abs(_hex_distance(position, self.zone_center) - nearest - 1),
                abs(position[1] - self.zone_center[1]),
                position,
            ),
        )
        chosen: list[Hex] = []
        for position in candidates:
            if len(chosen) == count:
                break
            if all(_hex_distance(position, other) >= self.ARCHER_GAP for other in chosen):
                chosen.append(position)
        return chosen

    def _assign(self, kind: str, posts: list[Hex], march_end: dict[tuple[str, int], Hex]) -> dict[tuple[str, int], Hex]:
        """Hand each post, in order, to the nearest slot still free at the end of the march.

        A slot left without a post holds where the march ended.
        """
        free = list(range(len(self.infantry[kind])))
        assigned: dict[tuple[str, int], Hex] = {}
        for post in posts:
            if not free:
                break
            slot_index = min(free, key=lambda index: (_hex_distance(march_end[(kind, index)], post), index))
            free.remove(slot_index)
            assigned[(kind, slot_index)] = post
        for slot_index in free:
            assigned[(kind, slot_index)] = march_end[(kind, slot_index)]
        return assigned

    def _plan_wings(self, cavalry: list[int], spawns: dict | None) -> None:
        """Split the cavalry into two even wings and give each rider a staging and a strike tile."""
        if spawns is not None:
            ordered = sorted(cavalry, key=lambda index: (spawns[("cavalry", index)][1], index))
        else:
            ordered = sorted(cavalry, key=lambda index: (index % 2, index))
        half = (len(ordered) + 1) // 2
        passages = self._passage_rows()
        center_q, center_r = self.zone_center
        reach = self.STRIKE_REACH
        wings = (
            # North: out from the zone's far north tile, toward the northeast.
            (ordered[:half], passages[0], (center_q + 1 + reach, center_r - 1 - reach)),
            # South: out from the zone's far south tile, toward the southeast.
            (ordered[half:], passages[-1], (center_q + 1, center_r + reach)),
        )
        self._wings: dict[int, tuple[int, Hex, Hex]] = {}
        for wing, (riders, passage_r, strike) in enumerate(wings):
            stage_tiles = self._cluster(self._onto_field(center_q - self.STAGE_DEPTH, passage_r), len(riders))
            strike_tiles = self._cluster(self._onto_field(*strike), len(riders))
            for slot_index, index in enumerate(riders):
                self._wings[index] = (wing, stage_tiles[slot_index], strike_tiles[slot_index])

    def _cluster(self, anchor: Hex, count: int) -> list[Hex]:
        """Return the ``count`` passable tiles nearest ``anchor``, the anchor itself first."""
        return sorted(self.terrain.entry_cost, key=lambda position: (_hex_distance(position, anchor), position))[:count]

    def _route(self, waypoints: list[Hex]) -> list[Hex]:
        """Join straight hex lines between waypoints into one route, skipping impassable tiles."""
        route = [waypoints[0]]
        for start, end in zip(waypoints, waypoints[1:]):
            for position in _hex_line(start, end)[1:]:
                if position != route[-1] and self.terrain.passable(position):
                    route.append(position)
        return route

    def _choose_rally(self, members: dict[str, list[int]], centroid: Hex) -> Hex:
        """Return the tile nearest the centroid where the whole spaced-out formation fits."""
        candidates = sorted(self.terrain.entry_cost, key=lambda position: (_hex_distance(position, centroid), position))
        for anchor in candidates:
            if _hex_distance(anchor, centroid) > self.RALLY_SEARCH:
                break
            footprint = [
                self._ideal_slot(members, kind, slot_index, anchor)
                for kind, indices in members.items()
                for slot_index in range(len(indices))
            ]
            if len(set(footprint)) == len(footprint) and all(self.terrain.passable(position) for position in footprint):
                return anchor
        return centroid

    def _march_schedule(self, route: list[Hex]) -> list[int]:
        """Walk the route the way a line troop would, spending ``pace`` movement points a round.

        As in the rules, the first step of a round is always allowed and later steps need enough
        points left. Entry ``k`` is the route index the anchor reaches after ``k`` rounds.
        """
        schedule = [0]
        index = 0
        while index < len(route) - 1:
            points = self.pace
            moved = False
            while index < len(route) - 1:
                cost = self.terrain.entry_cost.get(route[index + 1], _UNREACHABLE)
                if moved and cost > points:
                    break
                points -= cost
                index += 1
                moved = True
                if points <= 0:
                    break
            schedule.append(index)
        return schedule

    def _anchor(self, round_number: int) -> Hex:
        marched = max(0, round_number - self.ready_round)
        return self.route[self.schedule[min(len(self.schedule) - 1, marched)]]

    def _formation(self, members: dict[str, list[int]], anchor: Hex) -> dict[tuple[str, int], Hex]:
        """Return every slot of the infantry formation around ``anchor``, keyed by (type, slot index).

        Slots are placed footmen first, then archers, and a slot never takes a tile an earlier slot
        already holds.
        """
        taken: set[Hex] = set()
        slots: dict[tuple[str, int], Hex] = {}
        for kind in ("footman", "archer"):
            for slot_index in range(len(members.get(kind, []))):
                q, r = self._ideal_slot(members, kind, slot_index, anchor)
                position = self._settle(q, r, anchor[1], taken)
                taken.add(position)
                slots[(kind, slot_index)] = position
        return slots

    def _ideal_slot(self, members: dict[str, list[int]], kind: str, slot_index: int, anchor: Hex) -> Hex:
        """Footmen pack into a tight hex around ``anchor``, the same shape as the zone, so they
        stand shoulder to shoulder and slide into it together. Archers line up behind them.
        """
        front, middle = anchor
        if kind == "footman":
            dq, dr = _CLUSTER[slot_index % len(_CLUSTER)]
            return front + dq, middle + dr
        footmen = len(members.get("footman", []))
        depth = max((_hex_distance((0, 0), offset) for offset in _CLUSTER[:footmen]), default=0)
        count = len(members.get(kind, []))
        row = middle + slot_index * self.FILE_SPACING - ((count - 1) * self.FILE_SPACING) // 2
        return front - depth - self.RANK_SPACING, row

    def _settle(self, q: int, r: int, toward_r: int, taken: set[Hex] = frozenset()) -> Hex:
        """Slide a slot toward the anchor row, then back toward home, until it is on free ground.

        That keeps slots on the field near an edge, and squeezes a formation through tight ground,
        queueing the rest single file behind it.
        """
        start = (q, r)
        for _ in range(4 * self.side):
            if self.terrain.passable((q, r)) and (q, r) not in taken:
                return q, r
            if r != toward_r:
                r += 1 if r < toward_r else -1
            elif q > 0:
                q -= 1
            else:
                break
        return start if self.terrain.passable(start) else (q, r)

    def _onto_field(self, q: int, r: int) -> Hex:
        """Slide a waypoint toward the middle row until it is on passable ground."""
        return self._settle(q, r, self.extent)

    def _spawns(self, count: int) -> list[Hex]:
        """Rebuild red's spawn tiles exactly as the battlefield generator lays them out."""
        left = [(q, r) for q in range(self.extent) for r in range(self.side) if self.terrain.passable((q, r))]
        if len(left) < count:
            return []
        return [left[index * len(left) // count] for index in range(count)]

    def _passage_rows(self) -> list[int]:
        """Return the middle row of each gap in the river seam, from north to south."""
        runs: list[list[int]] = []
        for r in range(self.side):
            if self.terrain.passable((self.extent, r)):
                if runs and runs[-1][-1] == r - 1:
                    runs[-1].append(r)
                else:
                    runs.append([r])
        if not runs:
            return [self.extent]
        return [run[len(run) // 2] for run in runs]

    def _to_red(self, position: AxialPosition) -> Hex:
        if self.is_red:
            return position["q"], position["r"]
        return self.side - 1 - position["q"], self.side - 1 - position["r"]

    def _to_board(self, q: int, r: int) -> AxialPosition:
        if self.is_red:
            return {"q": q, "r": r}
        return {"q": self.side - 1 - q, "r": self.side - 1 - r}


# --------------------------------------------------------------------------------------------
# Per-activation context
# --------------------------------------------------------------------------------------------


class Context:
    """Everything a behavior block needs, computed once per activation."""

    def __init__(
        self,
        observation: SkirmishObservation,
        plan: ArmyPlan,
        terrain: TerrainMap,
        last_contact: int | None,
        released: bool = False,
        reports: list[tuple[str, Hex]] | None = None,
        advanced: bool = False,
        guard: tuple[Hex, list[tuple[str, Hex]]] | None = None,
        senders: list[Hex] | None = None,
    ) -> None:
        self.obs = observation
        self.here = me.position(observation)
        self.kind = me.unit_type(observation)
        self.index = int(me.unit_id(observation).rsplit("_", 1)[1])
        self.round = observation["observation"]["round"]
        self.enemies = visible.enemies(observation)
        self.allies = visible.allies(observation)
        self.nearest = (
            min(self.enemies, key=lambda enemy: tile.distance(self.here, enemy["position"])) if self.enemies else None
        )
        # The nearest enemy close enough to fight. Enemies seen farther off do not stop the march.
        self.threat = self.nearest if self.nearest is not None and self.nearest_distance <= ENGAGE_RADIUS else None
        self.nav = Navigator(observation, terrain)
        self.terrain = terrain
        self.plan = plan
        self.zone = plan.zone_board
        self.occupied = frozenset(_hex(unit["position"]) for unit in observation["observation"]["visible_units"])
        self.runner = plan.is_runner(self.kind, self.index)
        self.wing = plan.wing(self.index) if self.kind == "cavalry" else 0
        self.last_contact = last_contact
        # A rider stops waiting at its staging tile once it has seen an enemy, or been told of one.
        self.released = released
        # Enemies allies of the same type reported recently, as (type, board tile). They may have
        # moved since.
        self.reports = reports or []
        # A footman or archer that has seen the regroup finish heads for its post at once.
        self.advanced = advanced
        # For the one footman sent to defend an archer: that archer's tile and its attackers.
        self.guard = guard
        # Where the allies who sent recent reports were standing when they sent them.
        self.senders = senders or []

    @property
    def nearest_distance(self) -> int | None:
        return tile.distance(self.here, self.nearest["position"]) if self.nearest is not None else None

    def arrived(self) -> bool:
        """True once the infantry's regroup is over and units hold their posts."""
        return self.advanced or self.round > self.plan.arrive_round

    def scouting(self) -> bool:
        return self.plan.scouting(self.round, self.last_contact)

    def slot(self) -> AxialPosition:
        return self.plan.slot(self.kind, self.index, self.round, self.released, self.advanced)

    def allied(self, kind: str) -> list[AxialPosition]:
        """Positions of the visible allies of one unit type."""
        return [ally["position"] for ally in self.allies if ally["type"] == kind]

    def near_zone(self, position: AxialPosition) -> bool:
        """True for a tile inside the zone or touching it."""
        spot = _hex(position)
        return any(_hex_distance(spot, cell) <= 1 for cell in self.zone)

    def priority_targets(self) -> list[dict]:
        """Visible enemies, most urgent first: at the zone, then the weakest, then the nearest."""
        return sorted(
            self.enemies,
            key=lambda enemy: (
                not self.near_zone(enemy["position"]),
                enemy["hit_points"],
                tile.distance(self.here, enemy["position"]),
            ),
        )

    def target_from(self, landing: AxialPosition) -> tuple[int, dict | None]:
        """Return the rank and the most urgent enemy this unit can strike from ``landing``."""
        attack_range = units.STATS[self.kind].attack_range
        for rank, enemy in enumerate(self.priority_targets()):
            if tile.distance(landing, enemy["position"]) <= attack_range:
                return rank, enemy
        return len(self.enemies), None

    def order(self, path_id: int, target: dict | None = None) -> SkirmishAction:
        """Build an order along ``path_id`` naming ``target``, or the nearest visible enemy."""
        target = target or self.nearest
        if target is None:
            return action.move(path_id) if path_id else action.stay()
        if path_id:
            return action.move(path_id, target["unit_id"], self.obs)
        return action.stay(target["unit_id"], self.obs)


def _board(position: Hex) -> AxialPosition:
    return {"q": position[0], "r": position[1]}


# --------------------------------------------------------------------------------------------
# Behavior blocks
# --------------------------------------------------------------------------------------------


class Behavior:
    """A reusable decision rule. ``propose`` returns an order, or None when it does not apply."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        raise NotImplementedError


class WhenThreatened(Behavior):
    """Gate a combat block: it acts only once an enemy comes within ENGAGE_RADIUS."""

    def __init__(self, block: Behavior) -> None:
        self.block = block

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.threat is None:
            return None
        return self.block.propose(ctx)


class FightBack(Behavior):
    """Footman off the zone: close on an enemy within ``radius`` and strike it.

    The short leash keeps a marching footman from being picked apart without answering, while
    never drawing it far from its formation or its post.
    """

    def __init__(self, radius: int = 2) -> None:
        self.radius = radius

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.nearest is None or ctx.nearest_distance > self.radius:
            return None
        target = min(
            (enemy for enemy in ctx.enemies if tile.distance(ctx.here, enemy["position"]) <= self.radius),
            key=lambda enemy: (enemy["hit_points"], tile.distance(ctx.here, enemy["position"])),
        )
        goal = _hex(target["position"])
        footmen = [_hex(ally) for ally in ctx.allied("footman")]

        def key(landing: Hex) -> tuple[int, int]:
            # Next to the target first, then more adjacent footmen for the shield wall.
            buddies = sum(1 for ally in footmen if _hex_distance(ally, landing) == 1)
            return _hex_distance(landing, goal), -buddies

        return ctx.order(ctx.nav.best_path(key), target)


class KiteFromThreat(Behavior):
    """Back away from an enemy that is too close, unless the river keeps it from reaching us."""

    def __init__(self, radius: int = 2) -> None:
        self.radius = radius

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.nearest is None or ctx.nearest_distance > self.radius:
            return None
        # An enemy that needs more than two activations to walk around to us is no danger yet.
        reach = 2 * units.STATS[ctx.nearest["type"]].movement_points
        if ctx.nav.travel_cost(ctx.nearest["position"], ctx.here) > reach:
            return None
        return ctx.order(ctx.nav.path_away(ctx.nearest["position"]))


class HoldZone(Behavior):
    """Footman: once standing in the zone, never step off it. Strike from where it stands.

    The runner holds the zone center from the start. Everyone else holds whichever zone tile they
    reach once the march is over. The river seam walls the zone in above and below, so its far
    tiles can only be reached through its near ones: a holder steps forward onto an empty deeper
    zone tile beside it, so the zone fills from the front and those behind flow in after.
    With no enemy beside it, a holder moves to an empty zone tile that has one, so it can strike
    without ever leaving the zone. An enemy inside the zone is struck first, then the weakest one.
    """

    def propose(self, ctx: Context) -> SkirmishAction | None:
        here = _hex(ctx.here)
        if here not in ctx.zone:
            return None
        if not ctx.arrived():
            if not ctx.runner:
                return None
            # Before the others arrive, the runner keeps walking until it reaches a free center.
            center = _hex(ctx.slot())
            if here != center and center not in ctx.occupied:
                return None
        else:
            forward = [
                step
                for step in action.legal_steps(ctx.obs)
                if (landing := _hex(tile.at_path_end(ctx.here, step))) in ctx.zone
                and ctx.plan.depth(landing) > ctx.plan.depth(here)
            ]
            if forward:
                step = max(forward, key=lambda step: (ctx.plan.depth(_hex(tile.at_path_end(ctx.here, step))), -step))
                return ctx.order(step, ctx.target_from(tile.at_path_end(ctx.here, step))[1])
        adjacent = [enemy for enemy in ctx.enemies if tile.distance(ctx.here, enemy["position"]) == 1]
        if adjacent:
            return ctx.order(0, self._pick(ctx, adjacent))
        shift = self._shift(ctx)
        if shift is not None:
            return shift
        return ctx.order(0)

    @staticmethod
    def _pick(ctx: Context, adjacent: list[dict]) -> dict:
        return min(adjacent, key=lambda enemy: (_hex(enemy["position"]) not in ctx.zone, enemy["hit_points"]))

    def _shift(self, ctx: Context) -> SkirmishAction | None:
        """Move to an empty zone tile beside an enemy, or return None when there is none."""
        footmen = [_hex(ally) for ally in ctx.allied("footman")]
        best_path, best_key, best_target = None, None, None
        for path_id in action.legal_paths(ctx.obs):
            if path_id == 0:
                continue
            landing = tile.at_path_end(ctx.here, path_id)
            if _hex(landing) not in ctx.zone:
                continue
            adjacent = [enemy for enemy in ctx.enemies if tile.distance(landing, enemy["position"]) == 1]
            if not adjacent:
                continue
            target = self._pick(ctx, adjacent)
            buddies = sum(1 for ally in footmen if _hex_distance(ally, _hex(landing)) == 1)
            # An enemy inside the zone first, then the weakest, then the shield wall, then the
            # shortest walk.
            candidate = (
                _hex(target["position"]) in ctx.zone,
                -target["hit_points"],
                buddies,
                -len(paths.decode(path_id)),
            )
            if best_key is None or candidate > best_key:
                best_path, best_key, best_target = path_id, candidate, target
        if best_path is None:
            return None
        return ctx.order(best_path, best_target)


class DefendArcher(Behavior):
    """Footman sent from the zone: attack the enemies an archer reported attacking it.

    Enemies in sight near that archer come first. Otherwise the footman walks toward where the
    attackers were reported, or toward the archer itself.
    """

    def __init__(self, radius: int = 4) -> None:
        self.radius = radius

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.guard is None:
            return None
        archer, attackers = ctx.guard
        near = [enemy for enemy in ctx.enemies if _hex_distance(_hex(enemy["position"]), archer) <= self.radius]
        if near:
            target = min(near, key=lambda enemy: (tile.distance(ctx.here, enemy["position"]), enemy["hit_points"]))
            goal = _hex(target["position"])
            field = ctx.nav.field(goal)

            def key(landing: Hex) -> tuple[int, int]:
                return (0 if _hex_distance(landing, goal) == 1 else 1), field.get(landing, _UNREACHABLE)

            return ctx.order(ctx.nav.best_path(key), target)
        here = _hex(ctx.here)
        goal = min(
            (position for _, position in attackers),
            key=lambda position: _hex_distance(here, position),
            default=archer,
        )
        return ctx.order(ctx.nav.step_toward(_board(goal)))


class FillZoneGap(Behavior):
    """Footman off the zone after arrival: step onto an empty zone tile, or strike at its edge.

    This is how the reserve replaces a fallen zone footman, and how anyone who loses a race for
    a tile finds another one.
    """

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if not ctx.arrived():
            return None
        here = _hex(ctx.here)
        sight = units.STATS[ctx.kind].vision
        # Only tiles in sight are known to be empty.
        gaps = [cell for cell in ctx.zone if cell not in ctx.occupied and _hex_distance(cell, here) <= sight]
        if gaps:
            gap = min(gaps, key=lambda cell: (ctx.nav.travel_cost(ctx.here, _board(cell)), cell))
            path = ctx.nav.step_toward(_board(gap))
            if path:
                return ctx.order(path)

        intruders = [enemy for enemy in ctx.enemies if ctx.near_zone(enemy["position"])]
        if not intruders:
            return None
        target = min(intruders, key=lambda enemy: (enemy["hit_points"], tile.distance(ctx.here, enemy["position"])))
        goal = _hex(target["position"])

        def key(landing: Hex) -> tuple[int, int]:
            # Only fight from the zone's edge, so nobody gets drawn away from it.
            if not ctx.near_zone(_board(landing)):
                return 1, _UNREACHABLE
            return (0 if _hex_distance(landing, goal) == 1 else 1), _hex_distance(landing, goal)

        return ctx.order(ctx.nav.best_path(key), target)


class ClaimEmptyZone(Behavior):
    """Archer or cavalry: once no ally stands in the zone, step in so it keeps scoring.

    The first unit in becomes the holder and the rest see it and stay at their posts. A holder
    leaves again once a footman is back in the zone beside it.
    """

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if not ctx.arrived():
            return None
        here = _hex(ctx.here)
        holders = [ally for ally in ctx.allies if _hex(ally["position"]) in ctx.zone]
        if here in ctx.zone:
            if any(holder["type"] == "footman" for holder in holders):
                return None
            return ctx.order(0, ctx.target_from(ctx.here)[1])
        if holders:
            return None
        sight = units.STATS[ctx.kind].vision
        if any(_hex_distance(cell, here) > sight for cell in ctx.zone):
            return None  # the zone is not fully in sight, so it may not be empty
        cells = [cell for cell in ctx.zone if cell not in ctx.occupied]
        if not cells:
            return None
        cell = min(cells, key=lambda cell: (ctx.nav.travel_cost(ctx.here, _board(cell)), cell))
        path = ctx.nav.step_toward(_board(cell))
        if not path:
            return None
        return ctx.order(path, ctx.target_from(tile.at_path_end(ctx.here, path))[1])


class TakeHighGround(Behavior):
    """Archer: with an enemy in bow range of a free hill nearby, climb it and shoot from there.

    Only hills on our side of the river count, so the footmen still screen the archer, and none
    within ``safe`` tiles of an enemy. An archer already on such a hill stays on it.
    ``KiteFromThreat`` runs first, so an enemy that gets too close still drives it off.
    """

    def __init__(self, search: int = 4, safe: int = 2) -> None:
        self.search = search
        self.safe = safe

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if not ctx.enemies:
            return None
        here = _hex(ctx.here)
        attack_range = units.STATS[ctx.kind].attack_range
        enemies = [_hex(enemy["position"]) for enemy in ctx.enemies]

        def usable(hill: Hex) -> bool:
            return (
                ctx.plan.on_our_side(hill)
                and (hill == here or hill not in ctx.occupied)
                and any(_hex_distance(hill, enemy) <= attack_range for enemy in enemies)
                and all(_hex_distance(hill, enemy) > self.safe for enemy in enemies)
            )

        if here in ctx.terrain.hills and usable(here):
            return ctx.order(0, ctx.target_from(ctx.here)[1])

        # A hill counts as close when the archer can reach it within two activations.
        reach = 2 * units.STATS[ctx.kind].movement_points
        options = []
        for hill in ctx.terrain.hills:
            if _hex_distance(hill, here) > self.search or not usable(hill):
                continue
            cost = ctx.terrain.field(hill).get(here, _UNREACHABLE)
            if cost <= reach:
                options.append((cost, hill))
        if not options:
            return None
        hill = _board(min(options)[1])
        path = ctx.nav.step_toward(hill)
        if not path:
            return None
        return ctx.order(path, ctx.target_from(tile.at_path_end(ctx.here, path))[1])


class ShootFromBehindFootmen(Behavior):
    """Archer: shoot the most urgent enemy in reach from a tile screened by a footman."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        enemies = [enemy["position"] for enemy in ctx.enemies]
        footmen = ctx.allied("footman")

        best_path, best_key, best_target = None, None, None
        for path_id in action.legal_paths(ctx.obs):
            if path_id and ctx.nav.path_has_bad_terrain(path_id):
                continue
            landing = tile.at_path_end(ctx.here, path_id)
            if path_id and not ctx.plan.on_our_side(_hex(landing)):
                continue  # never cross the river ahead of the footmen
            rank, target = ctx.target_from(landing)
            if target is None:
                continue
            shot = tile.distance(landing, target["position"])
            closest_enemy = min(tile.distance(landing, enemy) for enemy in enemies)
            screened = any(
                tile.distance(landing, ally) <= 2 and tile.distance(ally, target["position"]) < shot for ally in footmen
            )
            # Screened and out of reach first, then the most urgent target, then as far from
            # enemies as the shot allows, then the shortest walk.
            candidate = (screened, closest_enemy > 2, -rank, closest_enemy, -len(paths.decode(path_id)))
            if best_key is None or candidate > best_key:
                best_path, best_key, best_target = path_id, candidate, target

        if best_path is None:
            return None  # nothing is in reach this turn, so keep moving with the group
        return ctx.order(best_path, best_target)


def _approach(ctx: Context, goal: AxialPosition, slack: int = 2) -> tuple[int, dict | None]:
    """Close on ``goal`` along the real route, striking on the way when that costs little.

    Among the landings at most ``slack`` route cost worse than the best approach, one beside an
    enemy wins, so the move still ends in a strike instead of wasting the turn. Returns the path
    and the enemy to name, or the plain best approach and None when no such landing exists.
    """
    field = ctx.nav.field(_hex(goal))
    approach = ctx.nav.step_toward(goal)
    best = field.get(_hex(tile.at_path_end(ctx.here, approach)), _UNREACHABLE)
    pick, pick_key, victim = None, None, None
    for path_id in action.legal_paths(ctx.obs):
        if path_id and ctx.nav.path_has_bad_terrain(path_id):
            continue
        landing = tile.at_path_end(ctx.here, path_id)
        cost = field.get(_hex(landing), _UNREACHABLE)
        if cost > best + slack:
            continue
        adjacent = [enemy for enemy in ctx.enemies if tile.distance(landing, enemy["position"]) == 1]
        if not adjacent:
            continue
        enemy = min(adjacent, key=lambda enemy: (enemy["type"] != "archer", enemy["hit_points"]))
        # Closest to the goal first, then a charge, then the weakest victim.
        candidate = (cost, tile.distance(ctx.here, landing) < 3, enemy["hit_points"])
        if pick_key is None or candidate < pick_key:
            pick, pick_key, victim = path_id, candidate, enemy
    if pick is None:
        return approach, None
    return pick, victim


class HuntArchers(Behavior):
    """Cavalry: dive on any enemy archer within two rides, else flank the threat, charging if possible.

    With ``archers_only`` set it ignores everything but archers. Until the infantry reaches the
    zone or the rider has seen an enemy, it only fights on our side of the river, so it does not
    ride into the whole enemy army unprovoked. An archer with ``guards`` or more enemy footmen or
    cavalry beside it is left alone, since diving on it would trade a rider for little.
    """

    def __init__(self, guards: int = 2, archers_only: bool = False) -> None:
        self.guards = guards
        self.archers_only = archers_only

    def propose(self, ctx: Context) -> SkirmishAction | None:
        reach = 2 * units.STATS[ctx.kind].movement_points
        arrived = ctx.arrived() or ctx.released
        fighters = [enemy["position"] for enemy in ctx.enemies if enemy["type"] != "archer"]
        archers = []
        for enemy in ctx.enemies:
            if enemy["type"] != "archer":
                continue
            if not arrived and not ctx.plan.on_our_side(_hex(enemy["position"])):
                continue
            if sum(1 for fighter in fighters if tile.distance(fighter, enemy["position"]) <= 1) >= self.guards:
                continue
            cost = ctx.nav.travel_cost(ctx.here, enemy["position"])
            if cost <= reach:
                archers.append((cost, enemy["hit_points"], enemy["unit_id"], enemy))
        if not archers and self.archers_only:
            return None
        target = min(archers)[-1] if archers else ctx.threat
        if target is None:
            return None
        if not archers and not arrived and not ctx.plan.on_our_side(_hex(target["position"])):
            return None
        goal = target["position"]

        best_path, best_key = None, None
        for path_id in action.legal_paths(ctx.obs):
            if path_id and ctx.nav.path_has_bad_terrain(path_id):
                continue
            landing = tile.at_path_end(ctx.here, path_id)
            if tile.distance(landing, goal) != 1:
                continue
            # A walk ending 3 or more tiles from the start earns the charge bonus.
            charging = tile.distance(ctx.here, landing) >= 3
            crowding = sum(
                1 for enemy in ctx.enemies if enemy is not target and tile.distance(landing, enemy["position"]) <= 1
            )
            candidate = (charging, -crowding)
            if best_key is None or candidate > best_key:
                best_path, best_key = path_id, candidate

        if best_path is None:
            # Out of reach this turn: ride the real route, around the river if needed, striking
            # whoever stands beside the way so the turn is not wasted.
            best_path, victim = _approach(ctx, ctx.nav.flank_goal(goal))
            return ctx.order(best_path, victim or target)
        return ctx.order(best_path, target)


class AnswerSignal(Behavior):
    """Go to a fight an ally of the same type reported, though the enemy is out of our own sight.

    Only reports within ``rides`` activations of walking count, so units far from the fight keep
    their own post. A reported archer comes first. A report of a tile this unit can see with no
    enemy on it is stale and ignored. With ``zone_only`` set, only enemies at the zone count, so a
    footman defends the zone instead of chasing.
    """

    def __init__(self, rides: int = 3, archers_only: bool = False, zone_only: bool = False) -> None:
        self.rides = rides
        self.archers_only = archers_only
        self.zone_only = zone_only

    def propose(self, ctx: Context) -> SkirmishAction | None:
        reach = self.rides * units.STATS[ctx.kind].movement_points
        sight = units.STATS[ctx.kind].vision
        seen = {_hex(enemy["position"]) for enemy in ctx.enemies}
        options = []
        for kind, position in ctx.reports:
            if self.archers_only and kind != "archer":
                continue
            if self.zone_only and not ctx.near_zone(_board(position)):
                continue
            if position not in seen and _hex_distance(_hex(ctx.here), position) <= sight:
                continue
            cost = ctx.nav.travel_cost(ctx.here, _board(position))
            if cost <= reach:
                options.append((kind != "archer", cost, position))
        if not options:
            return None
        goal = min(options)[2]
        path, victim = _approach(ctx, ctx.nav.flank_goal(_board(goal)))
        return ctx.order(path, victim) if path or victim else None


class JoinFight(Behavior):
    """Archer: walk toward a fight another archer reported, however far, until it is in bow range.

    The archer never crosses the river or steps into the zone. It heads for the tile on our side
    that has the reported enemy in bow range, or, when the enemy stands too far across the river
    for any such tile, the tile on our side that comes closest. A reported tile the archer can see
    with nobody on it is stale, so the archer heads for the archers who sent the reports instead,
    since the fight is around them.
    """

    def __init__(self, beside: int = 2) -> None:
        self.beside = beside  # how close to a reporting archer is close enough

    def propose(self, ctx: Context) -> SkirmishAction | None:
        here = _hex(ctx.here)
        sight = units.STATS[ctx.kind].vision
        seen = {_hex(enemy["position"]) for enemy in ctx.enemies}
        live = [
            position
            for _, position in ctx.reports
            if position in seen or _hex_distance(here, position) > sight
        ]
        if live:
            reach, targets = units.STATS[ctx.kind].attack_range, live
        elif ctx.senders:
            reach, targets = self.beside, ctx.senders
        else:
            return None

        # Route cost of walking from a tile to here, which is close to the cost of walking there.
        walk = ctx.terrain.field(here)

        def key(position: Hex) -> tuple[int, int]:
            short = min(max(0, _hex_distance(position, target) - reach) for target in targets)
            return short, walk.get(position, _UNREACHABLE)

        spots = [
            position
            for position in ctx.terrain.entry_cost
            if ctx.plan.on_our_side(position)
            and position not in ctx.zone
            and (position == here or position not in ctx.occupied)
        ]
        best = min(spots, key=key, default=here)
        if key(best) >= key(here):
            return None  # already as close as our side of the river allows
        path = ctx.nav.step_toward(_board(best))
        return ctx.order(path) if path else None


class ChaseNearest(Behavior):
    """Close on the nearest visible enemy along the real route, around the river if needed."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.nearest is None:
            return None
        return ctx.order(ctx.nav.step_toward(ctx.nearest["position"]))


class KeepFormation(Behavior):
    """Walk to this unit's tile in the plan for this round, or hold it once there.

    With ``scout`` set, the block steps aside once the plan says to scout, so a unit can go
    looking for enemies that never came to the zone.
    """

    def __init__(self, scout: bool = False) -> None:
        self.scout = scout

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if self.scout and ctx.scouting():
            return None
        slot = ctx.slot()
        if ctx.here == slot:
            return ctx.order(0)
        return ctx.order(ctx.nav.step_toward(slot))


class SweepField(Behavior):
    """With no enemy in sight, tour the field's six outer regions in turn to flush out stragglers.

    Each wing starts on the opposite side of the tour, and every unit moves on to the next
    region after ``dwell`` rounds, so a unit hiding in a corner cannot stall the match.
    """

    def __init__(self, dwell: int = 12, inset: int = 3) -> None:
        self.dwell = dwell
        self.inset = inset

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.enemies:
            return None
        center = tile.at_center(ctx.obs)
        reach = center["q"] - self.inset
        regions = [{"q": center["q"] + dq * reach, "r": center["r"] + dr * reach} for dq, dr in tile.DIRECTIONS.values()]
        goal = regions[(ctx.round // self.dwell + 3 * ctx.wing) % len(regions)]
        path = ctx.nav.step_toward(goal)
        return action.move(path) if path else None


class HoldPosition(Behavior):
    """Last resort: stay put and strike whatever is in range."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        return ctx.order(0)


# --------------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------------


class Agent:
    """Runs the first behavior block in this unit type's playbook that proposes an order."""

    # Blocks are stateless, so every unit can share these instances. KeepFormation still names
    # the nearest enemy, so units strike anything in range on the way to their tile.
    PLAYBOOK: dict[str, list[Behavior]] = {
        # Footmen never chase: they hold the zone, fill its gaps, answer enemies right beside them
        # or reported at the zone, and otherwise keep the plan. The one exception is the footman
        # sent from the back of the zone to defend an archer under attack.
        "footman": [
            DefendArcher(),
            HoldZone(),
            FillZoneGap(),
            FightBack(2),
            AnswerSignal(rides=2, zone_only=True),
            KeepFormation(),
            HoldPosition(),
        ],
        "archer": [
            KiteFromThreat(2),
            ClaimEmptyZone(),
            TakeHighGround(),
            WhenThreatened(ShootFromBehindFootmen()),
            # An archer out of the fight walks toward one another archer reported, however far,
            # until it has the enemy in bow range, so no archer is left out of a battle.
            JoinFight(),
            # An archer that has seen nobody for a while goes looking, so a lone enemy picking
            # off the zone's footmen from out of sight cannot do it forever.
            KeepFormation(scout=True),
            ChaseNearest(),
            SweepField(),
            HoldPosition(),
        ],
        # Cavalry rides at full speed to its staging and strike tiles, and attacks whenever an
        # enemy archer is within two rides or any enemy comes close. A rider waits at its passage
        # until the infantry reaches the zone, unless it sees an enemy or another rider reports
        # one first. Archers, seen or reported, come before anything else.
        "cavalry": [
            HuntArchers(archers_only=True),
            AnswerSignal(archers_only=True),
            HuntArchers(),
            AnswerSignal(),
            ClaimEmptyZone(),
            KeepFormation(scout=True),
            ChaseNearest(),
            SweepField(),
            HoldPosition(),
        ],
    }

    def __init__(self) -> None:
        self.terrain: TerrainMap | None = None
        self.plan: ArmyPlan | None = None
        self.last_contact: int | None = None
        self.released = False
        self.reports: list[tuple[int, str, Hex]] = []  # (round seen, type, board tile)
        self.senders: list[tuple[int, Hex]] = []  # (round sent, where the reporting ally stood)
        self.round = 0
        self.outbox: dict[str, str] = {}  # player id -> text to send after this activation
        self.peers: list[str] = []
        self.footmen: list[str] = []
        self.advanced = False
        self.distress: list[tuple[int, Hex, list[tuple[str, Hex]]]] = []  # (round, archer, attackers)
        self.guard: tuple[int, Hex, list[tuple[str, Hex]]] | None = None  # the call this footman answers
        self.guard_heard: int | None = None  # last round another footman said it went to defend

    def reset(self, seed, observation) -> None:
        # The routing map and the plan depend only on constant match knowledge and the opening
        # position, so both are built once here, outside the decision clock.
        self.terrain = TerrainMap(observation)
        self.plan = ArmyPlan(observation, self.terrain)
        self.last_contact = None
        self.released = False
        self.reports = []
        self.senders = []
        self.round = 0
        self.outbox = {}
        self.peers = self._peers(observation)
        self.footmen = self._of_type(observation, "footman")
        self.advanced = False
        self.distress = []
        self.guard = None
        self.guard_heard = None

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        if self.plan is None:
            self.terrain = TerrainMap(observation)
            self.plan = ArmyPlan(observation, self.terrain)

        self.round = observation["observation"]["round"]
        self.reports = [report for report in self.reports if self.round - report[0] <= self.REPORT_AGE]
        self.senders = [sender for sender in self.senders if self.round - sender[0] <= self.REPORT_AGE]
        enemies = visible.enemies(observation)
        kind = me.unit_type(observation)
        if not self.released and (enemies or self.reports):
            self.released = True
        if not self.advanced and kind in ("footman", "archer"):
            self.advanced = self._regrouped(observation, kind)
        self.outbox = {}
        if kind in self.SIGNALLERS and enemies:
            report = self._report(enemies, _hex(me.position(observation)))
            for player in self.peers:
                self.outbox[player] = report
        if kind == "archer":
            self._call_for_help(observation, enemies)
        if kind == "footman":
            self._answer_call(observation, enemies)
        ctx = Context(
            observation,
            self.plan,
            self.terrain,
            self.last_contact,
            self.released,
            [(kind, position) for _, kind, position in self.reports],
            self.advanced,
            (self.guard[1], self.guard[2]) if self.guard is not None else None,
            [position for _, position in self.senders],
        )
        if ctx.threat is not None:
            self.last_contact = ctx.round
            ctx.last_contact = ctx.round

        for block in self.PLAYBOOK.get(ctx.kind, [HoldPosition()]):
            order = block.propose(ctx)
            if order is not None:
                return order
        return action.stay()

    def _regrouped(self, observation: SkirmishObservation, kind: str) -> bool:
        """True once this unit can tell the regroup is over and it should head for its post.

        A footman sees the whole tight footman block from inside it: once every block tile holds a
        footman, the regroup is done. Every unit also follows a footman already standing in the zone,
        since that one has advanced, and a footman answers a report of an enemy at the zone.
        """
        plan = self.plan
        footmen = [
            ally
            for ally in visible.allies(observation)
            if ally["type"] == "footman" and int(ally["unit_id"].rsplit("_", 1)[1]) != plan.runner
        ]
        if any(_hex(ally["position"]) in plan.zone_board for ally in footmen):
            return True
        if kind != "footman" or plan.is_runner(kind, int(me.unit_id(observation).rsplit("_", 1)[1])):
            return False
        if any(
            any(_hex_distance(position, cell) <= 1 for cell in plan.zone_board) for _, _, position in self.reports
        ):
            return True
        block = plan.rally_tiles("footman")
        filled = {_hex(ally["position"]) for ally in footmen} | {_hex(me.position(observation))}
        return block <= filled

    # -- Defending the archers ---------------------------------------------------------------------

    DISTRESS_RADIUS = 4  # an enemy this close to an archer, on our half of the river, is a breakthrough
    GUARD_HOLD = 3  # rounds a defender stays out after the last call, while no attacker is in sight
    MIN_HOLDERS = 3  # footmen that must stand in the zone before one of them may leave it

    def _call_for_help(self, observation: SkirmishObservation, enemies: list[dict]) -> None:
        """Archer: tell the footmen where it is and who is on it, when an enemy has broken through."""
        here = _hex(me.position(observation))
        center_depth = self.plan.zone_center[0]
        attackers = [
            enemy
            for enemy in enemies
            if _hex_distance(here, _hex(enemy["position"])) <= self.DISTRESS_RADIUS
            and self.plan.depth(_hex(enemy["position"])) < center_depth
        ]
        if not attackers:
            return
        listed = ";".join(
            f"{enemy['type'][0]}{enemy['position']['q']},{enemy['position']['r']}" for enemy in attackers[:8]
        )
        call = f"H{self.round}@{here[0]},{here[1]}|{listed}"
        for player in self.footmen:
            self.outbox[player] = call

    def _answer_call(self, observation: SkirmishObservation, enemies: list[dict]) -> None:
        """Footman: decide whether to go defend an archer that called, or come back from doing so."""
        self.distress = [call for call in self.distress if self.round - call[0] <= self.REPORT_AGE]
        if self.guard is not None:
            if self.distress:
                self.guard = max(self.distress)
            archer = self.guard[1]
            attacker_near = any(
                _hex_distance(_hex(enemy["position"]), archer) <= self.DISTRESS_RADIUS for enemy in enemies
            )
            if not attacker_near and self.round - self.guard[0] > self.GUARD_HOLD:
                self.guard = None  # the fight is over: back to the zone
            return
        if not self.distress:
            return
        if self.guard_heard is not None and self.round - self.guard_heard <= self.GUARD_HOLD:
            return  # another footman already went
        here = _hex(me.position(observation))
        zone = self.plan.zone_board
        holders = sum(
            1 for ally in visible.allies(observation) if ally["type"] == "footman" and _hex(ally["position"]) in zone
        )
        if here not in zone or holders + 1 < self.MIN_HOLDERS:
            return
        call = max(self.distress)
        # Only the footman on the back tile nearest the archer goes, so only one ever leaves.
        chosen = min(self.plan.back_tiles(), key=lambda cell: (_hex_distance(cell, call[1]), cell))
        if here != chosen:
            return
        self.guard = call
        notice = f"G{self.round}"
        for player in self.peers:
            self.outbox[player] = f"{self.outbox[player]} {notice}" if player in self.outbox else notice

    # Optional: a reinforcement-learning hook called after every step with that step's
    # transition. Its time counts against the timing and episode budget. The order argument is
    # what act returned. It is named order so it does not shadow the action helpers.
    #
    # def learn(self, observation, order: SkirmishAction, reward: float, terminated: bool) -> None:
    #     ...

    # -- Messaging: every unit reports the enemies it sees to allies of its own type -------------
    #
    # A report reads "14@9,12|a15,8;c16,9": in round 14 the sender, standing at (9,12), saw an
    # archer at (15,8) and a cavalry at (16,9), all in board coordinates. It reaches an ally at its next activation, after that ally has already
    # chosen its order, so it acts on the report one activation later still. A unit writes reports
    # only to allies of its own type, and never as a broadcast, since that reaches the enemy too.
    #
    # Two more messages break that rule on purpose. An archer under attack calls the footmen with
    # "H14@8,12|c9,13": in round 14 the archer at (8,12) has a cavalry on it at (9,13). The footman
    # that goes to help tells the other footmen "G14", so nobody else leaves the zone. One message
    # may carry several parts separated by spaces, since each ally gets one message per activation.

    SIGNALLERS = ("cavalry", "footman", "archer")  # unit types that send reports to their own kind

    REPORT_AGE = 2  # rounds a report stays worth acting on, since the enemy keeps moving
    REPORT_LIMIT = 12  # enemies per report, archers first, to stay inside the text limit

    def chat(self, inbox: list[dict]) -> list[dict] | None:
        for message in inbox:
            for part in message.get("text", "").split(" "):
                if part.startswith("H"):
                    call = self._parse_call(part)
                    if call is not None:
                        self.distress.append(call)
                elif part.startswith("G"):
                    try:
                        self.guard_heard = max(self.guard_heard or 0, int(part[1:]))
                    except ValueError:
                        pass
                else:
                    self.reports.extend(self._parse(part))
                    sender = self._parse_sender(part)
                    if sender is not None:
                        self.senders.append(sender)
        if not self.outbox:
            return None
        outbox, self.outbox = self.outbox, {}
        return [{"to": player, "text": text[:200]} for player, text in outbox.items()]

    @staticmethod
    def _parse_sender(text: str) -> tuple[int, Hex] | None:
        try:
            round_text, where = text.split("|", 1)[0].split("@")
            q, r = where.split(",")
            return int(round_text), (int(q), int(r))
        except ValueError:
            return None

    @classmethod
    def _parse_call(cls, part: str) -> tuple[int, Hex, list[tuple[str, Hex]]] | None:
        try:
            head, listed = part[1:].split("|", 1)
            round_text, where = head.split("@")
            q, r = where.split(",")
            attackers = [(kind, position) for _, kind, position in cls._parse(f"{round_text}|{listed}")]
            return int(round_text), (int(q), int(r)), attackers
        except ValueError:
            return None

    def _report(self, enemies: list[dict], here: Hex) -> str:
        ordered = sorted(enemies, key=lambda enemy: (enemy["type"] != "archer", enemy["hit_points"]))
        spotted = ";".join(
            f"{enemy['type'][0]}{enemy['position']['q']},{enemy['position']['r']}"
            for enemy in ordered[: self.REPORT_LIMIT]
        )
        return f"{self.round}@{here[0]},{here[1]}|{spotted}"

    @staticmethod
    def _parse(text: str) -> list[tuple[int, str, Hex]]:
        kinds = {"f": "footman", "a": "archer", "c": "cavalry"}
        try:
            round_text, spotted = text.split("|", 1)
            seen = int(round_text.split("@")[0])
            reports = []
            for entry in spotted.split(";"):
                q, r = entry[1:].split(",")
                reports.append((seen, kinds[entry[0]], (int(q), int(r))))
            return reports
        except (ValueError, KeyError, IndexError):
            return []  # not one of our reports

    @classmethod
    def _peers(cls, observation: SkirmishObservation) -> list[str]:
        """Player ids of every other ally of this unit's own type. Messages to the fallen are dropped."""
        return cls._of_type(observation, me.unit_type(observation))

    @staticmethod
    def _of_type(observation: SkirmishObservation, kind: str) -> list[str]:
        """Player ids of every other ally of one unit type."""
        own = me.unit_id(observation)
        return [
            entry["player"] for entry in roster.allies(observation) if entry["type"] == kind and entry["unit_id"] != own
        ]
