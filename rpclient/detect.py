import cv2 as cv
import numpy as np


def separate_rect(r, w, h, margin=0.1, eps=0.2):
    rects_out = []
    x0, y0 = r[0], r[1]
    r_w, r_h = r[2], r[3]

    eps_px = int(w * eps)
    margin_px = int(w * margin)

    w_n = int(np.ceil(r_w / w))
    h_n = int(np.ceil(r_h / w))
    if abs(r_w / w - w_n) < eps_px and r_w/r_h > 1 and w_n >= 2:
        y = int(y0 + r_h / 2 - h / 2)
        for i in range(int((r_w + 0.25*w) // w)):
            x = x0 + (w + margin_px) * i
            rects_out.append((x, y, w, h))

    elif abs(r_h / w - h_n) < eps_px and r_w/r_h < 1 and h_n >= 2:
        x = int(x0 + r_w / 2 - h / 2)
        for i in range(int((r_h + 0.25*w) // w)):
            y = y0 + (w + margin_px) * i
            rects_out.append((x, y, h, w))

    else:
        rects_out.append(r)

    return rects_out


def adj_contrast(src):
    vmin = np.percentile(src, 2)
    vmax = np.percentile(src, 98)
    src = np.clip(src, vmin, vmax).astype(np.float32)
    src = (src - vmin) / (vmax - vmin) * 255
    return src.astype(np.uint8)


def sorted_counterclockwise(pts):
    pts = np.array(pts)
    cx, cy = np.mean(pts, axis=0, dtype=np.float32)
    eps = 1e-4
    angles = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        if abs(dx) < eps:
            phi = np.pi/2 if dy > 0 else -np.pi/2
        else:
            phi = np.arctan(dy / dx)
            if dx < 0:
                phi += np.pi
        angles.append(phi)

    ics = np.argsort(angles)
    return pts[ics]


def simplify_contour(contour, n_corners=4):
    n_iter, max_iter = 0, 100
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return contour

        k = (lb + ub)/2.
        eps = k*cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub)/2.
        elif len(approx) < n_corners:
            ub = (lb + ub)/2.
        else:
            return approx


def crop_transform_road(src, white_thresh=150):
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    im = cv.medianBlur(src_gray, 5)

    _, thresh = cv.threshold(im, white_thresh, 255, cv.THRESH_BINARY)

    ctrs, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    areas = [cv.contourArea(c) for c in ctrs]
    sheet_idx = np.argmax(areas)

    poly_ctr = simplify_contour(ctrs[sheet_idx])
    if len(poly_ctr) > 4:
        print(len(poly_ctr))
        print('''
              ERROR: simplified contour has more than 4 points.
              This probably means that findContour failed to find
              the sheet contour.
              ''')

        # Possible solutions
        # 1. Clip the contour. Risky.
        # 2. Try various thresholds using binsearch. This might work.
        # 3. It's also probable that sheet might not have the largest contour,
        #    so it might be a good idea to get the first largest contour
        #    that looks like a trapeeze (rectangle).
        # TODO: try these fixes
        return None
    
    r = cv.boundingRect(poly_ctr)
    pts1 = np.float32(sorted_counterclockwise(poly_ctr.reshape((4, 2))))
    pts2 = np.float32([[r[0], r[1]], [r[0], r[1] + r[3]], 
                       [r[0] + r[2], r[1] + r[3]], [r[0] + r[2], r[1]]])
    pts2 = sorted_counterclockwise(pts2)

    M = cv.getPerspectiveTransform(pts1, pts2)
    warped = cv.warpPerspective(src, M, (src.shape[1], src.shape[0]))
    warped_cropped = warped[r[1]:r[1]+r[3], r[0]:r[0]+r[2], :]

    return warped_cropped


def find_cars(src, area_min, area_max, ratio_min, ratio_max, car_w, car_h):
    if area_max is None:
        area_max = src.shape[1] * src.shape[0]

    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.medianBlur(src_gray, 11)
    src_gray = adj_contrast(src_gray)

    edges = cv.Canny(src_gray, 150, 200)
    edges = cv.dilate(edges, np.ones((5,5)), iterations=1)

    ctrs, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    car_rects = []
    areas = []
    for c in ctrs:
        area = cv.contourArea(c)
        r = cv.boundingRect(c)
        ratio = max(r[2], r[3]) / min(r[2], r[3])

        areas.append(area)
        if area < area_min or area > area_max:
            continue
        if ratio < ratio_min or ratio > ratio_max:
            continue
        # TODO: test whether the contour looks like a rect
        # using something like approxPolyDP with small eps

        car_rects.extend(separate_rect(r, car_w, car_h))
        center = (r[0] + r[2]//2, r[1] + r[3]//2)

    print(np.max(areas), area_min, area_max)

    return car_rects, edges


def detect(src, ret_all=False):
    cim = crop_transform_road(src)
    if cim is None:
        return None

    w, h = cim.shape[1], cim.shape[0]
    q = w // 10

    # car_w, car_h = int(0.75 * q), int(q * 0.4)
    car_w, car_h = int(0.66 * q), int(0.33 * q)
    area_min, area_max = 0.7 * car_w*car_h, 4 * car_w*car_h
    # area_min, area_max = 0.2* q**2, 1.5 * q**2
    ratio_min, ratio_max = 1, 4 

    car_rects, im = find_cars(cim, area_min, area_max, 
            ratio_min, ratio_max, car_w, car_h)

    HOR, VERT = 0, 1
    segments = [
        (HOR, h/2 - q/2, w/2, w), #0
        (VERT, w/2 + q/2, 0, h/2), #1
        (VERT, w/2 - q/2, 0, h/2), #2
        (HOR, h/2 - q/2, 0, w/2), #3
        (HOR, h/2 + q/2, 0, w/2), #4
        (VERT, w/2 - q/2, h/2, h), #5
        (VERT, w/2 + q/2, h/2, h), #6
        (HOR, h/2 + q/2, w/2, w), #7
    ]

    # TODO: refactor me! better variable names
    cars = []
    for r in car_rects:
        ratio = r[2] / r[3]
        x, y = r[0] + r[2] // 2, r[1] + r[3] // 2

        orient = HOR if ratio > 1 else VERT
        u, v = (x, y) if orient == HOR else (y, x)
        min_d = max(w, h)
        lane_idx = -1
        for i, s in enumerate(segments):
            s_orient, pos, left, right = s
            if s_orient != orient or u < left or u > right:
                continue

            d = abs(pos - v)
            if d < min_d:
                min_d = d
                lane_idx = i

        print(f'({x}, {y}) -> {lane_idx}')
        cars.append(((x, y), ratio, lane_idx))

    print('***', len(car_rects))
    cars.sort(key=lambda c: c[0][1] * w + c[0][0])

    if ret_all:
        return cars, w, h, cim, im, car_rects
    else:
        return cars, w, h


def main():
    # WARNING: I deleted a bunch of redundant code. I hope nothing's broken lmao.
    # TODO: test

    result = detect(cv.imread('frames/8.jpg'), ret_all=True)
    cars, w, h, cim, im, car_rects = result
    q = w // 10

    cv.line(cim, (0, h // 2), (w, h // 2), (255, 0, 0), 1)
    cv.line(cim, (0, h // 2 - q), (w, h // 2 - q), (0, 0, 255), 3)
    cv.line(cim, (0, h // 2 + q), (w, h // 2 + q), (0, 0, 255), 3)

    cv.line(cim, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
    cv.line(cim, (w // 2 - q, 0), (w // 2 - q, h), (0, 0, 255), 3)
    cv.line(cim, (w // 2 + q, 0), (w // 2 + q, h), (0, 0, 255), 3)

    for r in car_rects:
        cv.rectangle(cim, r, (0, 255, 255), 3)

    cv.imshow('cropped_transformed', cv.resize(cim, None, fx=0.5, fy=0.5))
    cv.imshow('im', cv.resize(im, None, fx=0.5, fy=0.5))
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

