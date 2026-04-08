import numpy as np

GAME_WIDTH = 384
GAME_HEIGHT = 448

def closest_point(points, point):
    if points.size == 0:
        return np.array([-1, -1, 0, 0], dtype=np.float32)

    # Mask out padded -1 entries
    mask = points[:, 0] != -1
    if not np.any(mask):
        return np.array([-1, -1, 0, 0], dtype=np.float32)

    valid_points = points[mask]
    px, py = point
    dx = valid_points[:, 0] - px
    dy = valid_points[:, 1] - py
    distances = np.sqrt(dx**2 + dy**2)
    idx = np.argmin(distances)

    closest = valid_points[idx]
    if points.shape[1] >= 4:
        return np.array([closest[0], closest[1], closest[2], closest[3]], dtype=np.float32)
    else:
        return np.array([closest[0], closest[1], 0, 0], dtype=np.float32)

def item_intersects_hitbox(player_x, player_y, hitbox, item_x, item_y, max_distance=448):
    x1, x2 = player_x - hitbox, player_x + hitbox

    # Check if item's x is within the player's horizontal hitbox
    if not (x1 <= item_x <= x2) or (player_y < item_y):
        return False, -1

    distance = np.sqrt((item_x - player_x) ** 2 + (item_y - player_y) ** 2)
    if distance > max_distance:
        return False, -1

    return True, distance


def bullet_intersects_hitbox(player_x, player_y, hitbox, bullet_data, max_distance=590):
    x1, x2 = player_x - hitbox, player_x + hitbox
    y1, y2 = player_y - hitbox, player_y + hitbox

    bx = bullet_data[:, 0]
    by = bullet_data[:, 1]
    dx = bullet_data[:, 2]
    dy = bullet_data[:, 3]

    epsilon = 1e-8
    dx = np.where(dx == 0, epsilon, dx)
    dy = np.where(dy == 0, epsilon, dy)

    tx1 = (x1 - bx) / dx
    tx2 = (x2 - bx) / dx
    ty1 = (y1 - by) / dy
    ty2 = (y2 - by) / dy

    tmin_x = np.minimum(tx1, tx2)
    tmax_x = np.maximum(tx1, tx2)
    tmin_y = np.minimum(ty1, ty2)
    tmax_y = np.maximum(ty1, ty2)

    t_entry = np.maximum(tmin_x, tmin_y)
    t_exit = np.minimum(tmax_x, tmax_y)

    dists = np.sqrt((bx - player_x)**2 + (by - player_y)**2)
    valid = (t_exit >= 0) & (t_entry <= t_exit) & (dists <= max_distance)

    return valid, dists

def get_entities(entities, m=100):
    return entities[:m] + [None] * max(0, m - len(entities))

def get_boss(boss):
    if boss:
        return boss.x / GAME_WIDTH, boss.y / GAME_HEIGHT
    else:
        return -1, -1
