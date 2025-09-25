def detect_circle(gray_image: np.ndarray, binary_image: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Detect the main circle (clock boundary) with stricter validation."""
    h, w = gray_image.shape[:2]
    min_dim = min(h, w)

    # --- Step 1: HoughCircles attempt ---
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min_dim / 4),           # bigger spacing between possible circles
        param1=120,
        param2=40,                          # higher threshold = fewer false positives
        minRadius=int(min_dim * 0.25),      # at least 25% of canvas
        maxRadius=int(min_dim * 0.48)
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # pick largest
        cx, cy, r = sorted(circles, key=lambda c: c[2], reverse=True)[0]
        return (int(cx), int(cy), int(r))

    # --- Step 2: Contour-based fallback ---
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_circle = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:   # ignore tiny shapes
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < min_dim * 0.2:   # must be big enough
            continue
        circle_area = math.pi * (radius ** 2)
        if circle_area <= 0:
            continue
        circularity = area / circle_area
        # only keep if fairly round
        if circularity < 0.65:
            continue
        score = circularity * radius
        if score > best_score:
            best_score = score
            best_circle = (int(x), int(y), int(radius))

    return best_circle
