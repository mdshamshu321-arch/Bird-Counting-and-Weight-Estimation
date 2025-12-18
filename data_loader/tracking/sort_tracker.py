import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
        (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh
    )
    return o


class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.id = track_id
        self.age = 0
        self.hits = 1

    def update(self, bbox):
        self.bbox = bbox
        self.age = 0
        self.hits += 1

    def predict(self):
        self.age += 1
        return self.bbox


class Sort:
    def __init__(self, max_age=5, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        updated_tracks = []

        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(det[:4], self.next_id))
                self.next_id += 1
            return self._get_tracks()

        iou_matrix = np.zeros((len(self.tracks), len(detections)))

        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = iou(track.bbox, det[:4])

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched, unmatched_tracks, unmatched_dets = [], [], []

        for t in range(len(self.tracks)):
            if t not in row_ind:
                unmatched_tracks.append(t)

        for d in range(len(detections)):
            if d not in col_ind:
                unmatched_dets.append(d)

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                self.tracks[r].update(detections[c][:4])
                matched.append(r)
            else:
                unmatched_tracks.append(r)
                unmatched_dets.append(c)

        for d in unmatched_dets:
            self.tracks.append(Track(detections[d][:4], self.next_id))
            self.next_id += 1

        for t in unmatched_tracks:
            self.tracks[t].predict()

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        return self._get_tracks()

    def _get_tracks(self):
        outputs = []
        for track in self.tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            outputs.append([x1, y1, x2, y2, track.id])
        return np.array(outputs)
