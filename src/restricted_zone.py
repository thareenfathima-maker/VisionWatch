def is_inside_zone(point, zone):
    """Return True if a point is inside a rectangular zone."""
    x, y = point
    x1, y1, x2, y2 = zone

    return x1 <= x <= x2 and y1 <= y <= y2