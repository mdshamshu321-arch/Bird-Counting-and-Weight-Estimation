# Bird-Counting-and-Weight-Estimation
## Task Objective

This project implements a computer vision prototype that processes fixed-camera poultry CCTV footage to produce:

1. **Bird Counts Over Time** using object detection and stable tracking IDs to avoid double counting across frames.

2. **Bird Weight Estimation** using a defined visual size-based proxy derived from detected bounding box characteristics, intended as a relative weight index in the absence of ground-truth measurements.

This repository presents a computer vision prototype designed to analyze fixed-camera
poultry CCTV footage for automated bird monitoring. The system detects and tracks birds
across video frames to produce reliable bird counts over time and estimates bird weight
using a visual size-based proxy.

The solution integrates object detection and tracking to handle crowded flock scenarios
and supports scalable weight calibration when real-world reference data is available.
This project is intended as a proof-of-concept for livestock analytics and monitoring
applications.

## Key Capabilities
- Bird detection from fixed-camera CCTV footage
- Stable tracking of birds across frames
- Temporal bird counting using tracked identities
- Relative weight estimation based on visual size cues
- Extendable design for real-world weight calibration

---
<img width="2180" height="1536" alt="poultry-detection png" src="https://github.com/user-attachments/assets/fa153b58-748c-47b4-a99d-3fa38d6f274d" />



