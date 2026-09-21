import numpy as np

GAME_WIDTH = 384
GAME_HEIGHT = 448

def entity_array(entities, max_entities, n_features, row_fn):
    arr = np.zeros((max_entities, n_features), dtype=np.float32)
    rows = [row_fn(e) for e in entities[:max_entities] if e]
    if rows:
        arr[:len(rows)] = np.clip(np.array(rows, dtype=np.float32), -1, 1)
    return arr

def closest_entity(entities):
    present = entities[:, 0] > 0
    if not np.any(present):
        return np.zeros(entities.shape[1], dtype=np.float32)
    valid = entities[present]
    return valid[np.argmin(valid[:, 1] ** 2 + valid[:, 2] ** 2)]

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

def get_onscreen_entities(entities, m=100):
    visible = [e for e in entities if e and 0 <= e.x <= GAME_WIDTH and 0 <= e.y <= GAME_HEIGHT]
    return visible[:m] + [None] * max(0, m - len(visible))
