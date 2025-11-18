# Part 1 – Object Detection Using Template Matching (Correlation)

## Problem Description

In this part of the assignment we implement **object detection using template matching through the correlation method**.

- The template image is **not** cropped from the test image.  
- Each template is taken from a **completely different scene**.  
- Detection is evaluated on **10 different objects** (possibly appearing in the same or in different images).  

---

## Template Images (Examples)

Below are some of the template images used for correlation-based matching:

<p float="left">
  <img src="Part-1/templates/Brain.jpg" width="160" />
  <img src="Part-1/templates/Mouse.jpg" width="160" />
  <img src="Part-1/templates/Key.jpg" width="160" />
  <img src="Part-1/templates/Measure.jpg" width="160" />
</p>

Additional templates (not shown here) are used so that the total number of distinct objects evaluated is 10.

---

## Scene Images (Examples)

Some of the scene images where objects are searched for:

<p float="left">
  <img src="Part-1/scenes/5819145774133087418.jpg"   width="160" />
  <img src="Part-1/scenes/5819145774133087421.jpg"   width="160" />
  <img src="Part-1/scenes/5819145774133087422.jpg"     width="160" />
  <img src="Part-1/scenes/5819145774133087426.jpg" width="160" />
</p>


Example visualization:

<img src="Part-1/scenes/5819145774133087426.jpg" width="400" />

---

## Detection Results (Examples)

Below are a few sample detection outputs for different template–scene pairs  
(templates are matched using correlation; detected locations are highlighted in the result images):



<p float="left">
  <img src="Part-1/results/orb_Ball.jpg_5819145774133087422.jpg" width="220" />
  <img src="Part-1/results/orb_Key.jpg_5819145774133087426.jpg" width="220" />
  <img src="Part-1/results/orb_Measure.jpg_5819145774133087426.jpg" width="220" />
  <img src="Part-1/results/orb_Measure.jpg_5819145774133087428.jpg" width="220" />
</p>

These examples illustrate successful localization of the objects in scenes where **the template and scene come from different original photographs**, satisfying the assignment requirement.
