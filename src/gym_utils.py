import numpy as np

offset_x = 32
offset_y = 16
x = 384
y = 448

def closest_point(points, point):
    if points.size == 0:
        return -1, -1, -1, -1, -1, -1, -1  # Default value if no objects exist

    px, py = point
    min_distance_sq = np.inf
    closest = None

    for p in points:
        n_x = p[0]
        n_y = p[1]
        dx = n_x - px
        dy = n_y - py
        distance_sq = np.sqrt(dx ** 2 + dy ** 2)
        if distance_sq < min_distance_sq:
            min_distance_sq = distance_sq
            closest = (n_x, n_y, distance_sq, p[2], p[3], p[4], p[5])

    # Normalize coordinates and distance
    if closest:
        return closest[0], closest[1], closest[2], closest[3], closest[4], closest[5], closest[6]  # Normalize

    return -1, -1, -1, -1, -1, -1, -1  # Fallback in case of error

def item_intersects_hitbox(player_x, player_y, hitbox, item_x, item_y, max_distance=448):
    x1, x2 = player_x - hitbox, player_x + hitbox

    # Check if item's x is within the player's horizontal hitbox
    if not (x1 <= item_x <= x2) or (player_y < item_y):
        return False, -1

    distance = np.sqrt((item_x - player_x) ** 2 + (item_y - player_y) ** 2)
    if distance > max_distance:
        return False, -1

    return True, distance


def bullet_intersects_hitbox(player_x, player_y, hitbox, bullet_x, bullet_y, dx, dy, max_distance=590):
    x1, x2 = player_x - hitbox, player_x + hitbox
    y1, y2 = player_y - hitbox, player_y + hitbox

    epsilon = 1e-8
    dx = dx if dx != 0 else epsilon
    dy = dy if dy != 0 else epsilon

    tx1 = (x1 - bullet_x) / dx
    tx2 = (x2 - bullet_x) / dx
    ty1 = (y1 - bullet_y) / dy
    ty2 = (y2 - bullet_y) / dy

    # Ensure correct min/max intervals
    tmin_x, tmax_x = min(tx1, tx2), max(tx1, tx2)
    tmin_y, tmax_y = min(ty1, ty2), max(ty1, ty2)

    # Find global entry and exit points
    t_entry = max(tmin_x, tmin_y)
    t_exit = min(tmax_x, tmax_y)

    if t_exit < 0:
        return False, -1
    if t_entry > t_exit:
        return False, -1
    if t_entry > max_distance:
        return False, -1

    return True, np.sqrt((bullet_x - player_x) ** 2 + (bullet_y - player_y) ** 2)

def get_entities(entities, m=100):
    result = []
    for i in range(m):
        if i > len(entities) - 1:
            result.append(None)
        else:
            result.append(entities[i])
    return result

def get_boss(boss):
    if boss:
        return boss.x / x, boss.y / y
    else:
        return -1, -1
