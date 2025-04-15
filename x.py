import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import os

# --- Helper Functions ---

def detect_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return edges

def detect_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)
    return lines

def compute_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    det = A1 * B2 - A2 * B1
    if det == 0:
        return None
    else:
        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
        return (int(x), int(y))

def get_all_intersections(lines):
    points = []
    for line1, line2 in combinations(lines, 2):
        pt = compute_intersection(line1, line2)
        if pt is not None and all(0 <= v <= 5000 for v in pt):
            points.append(pt)
    return points

def estimate_vanishing_point(points):
    if len(points) == 0:
        return None
    points_np = np.array(points)
    x = int(np.median(points_np[:, 0]))
    y = int(np.median(points_np[:, 1]))
    return (x, y)

def draw_results(img, lines, vp):
    result = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if vp is not None:
        cv2.circle(result, vp, 8, (0, 0, 255), -1)
        cv2.putText(result, f"VP: {vp}", (vp[0]+10, vp[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return result

# --- Process Folder and Show All Images ---

def process_and_display_all(folder_path, max_images=4):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    image_files = image_files[:max_images]

    results = []
    titles = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        edges = detect_edges(img)
        lines = detect_lines(edges)

        if lines is None:
            print(f"No lines detected in {image_file}")
            results.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            titles.append(image_file + " (no lines)")
            continue

        intersections = get_all_intersections(lines)
        vp = estimate_vanishing_point(intersections)
        result = draw_results(img, lines, vp)
        results.append(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        titles.append(image_file)

    # Display all results in a grid
    plt.figure(figsize=(16, 8))
    for i, (img, title) in enumerate(zip(results, titles)):
        plt.subplot(1, len(results), i + 1)
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- Run ---

if __name__ == "__main__":
    folder_path = "/Users/saivenkatr/Downloads/Estimate_vanishing_points_data"  # Replace with your folder
    process_and_display_all(folder_path)
