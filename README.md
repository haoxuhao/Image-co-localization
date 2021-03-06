# Image co-localization
This repo is initially based on the [Deep-Descriptor-Transforming](https://github.com/FancyYan123/Deep-Descriptor-Transforming). 

### New features
- Spherical cluster method added [paper](https://www.jstage.jst.go.jp/article/mta/7/1/7_2/_article/-char/ja/).
- Test and evaluation script added.
- VOC & Object discovery dataset preprocessing.

***
# Visualizations

[DDT](https://www.researchgate.net/publication/316780426_Deep_Descriptor_Transforming_for_Image_Co-Localization)
-
<table>
    <tr>
        <td ><center><img src="./data/result/car-ddt/0.jpg" ></center></td>
        <td ><center><img src="./data/result/car-ddt/1.jpg"  ></center></td>
		<td ><center><img src="./data/result/car-ddt/2.jpg"  ></center></td>
		<td ><center><img src="./data/result/car-ddt/3.jpg"  ></center></td>
    </tr>
</table>


DDTplus:
-
<table>
    <tr>
        <td ><center><img src="./data/result/car-ddtplus/0.jpg" ></center></td>
        <td ><center><img src="./data/result/car-ddtplus/1.jpg"  ></center></td>
		<td ><center><img src="./data/result/car-ddtplus/2.jpg"  ></center></td>
		<td ><center><img src="./data/result/car-ddtplus/3.jpg"  ></center></td>
    </tr>
</table>

# CorLoc
## Object discovery

|impl | Airplane |  Car  | Horse |  Mean |
|--|--|--|--|--|
|here(DDT)| 82.9   | 70.8 |  81.7 | 78.5  |
| paper |  91.46   | 95.51 | 77.42 | 88.13 |

## VOC 07
|impl| aeroplane | bicycle |  bird |  boat | bottle |  bus  |  car  |  cat  | chair |  cow  | diningtable |  dog  | horse | motorbike | person | pottedplant | sheep |  sofa | train | tvmonitor |  Mean |
|-----------|---------|-------|-------|--------|-------|-------|-------|-------|-------|------|------|------|------|------|------|------|------|------|------|------|----|
|here(DDT)|   61.4   |   52.2  | 64.2 | 16.3 |  23.9  | 61.5 | 66.5 | 87.0 |  18.9 | 54.0 |     22.4    | 74.0 |  76.2 |    68.4   |  39.7  |     19.4    |  30.9 | 47.2 |  68.8 |    23.4   | 48.8 |
|paper | 67.3 | 63.3 | 61.3|22.7| 8.5 | 64.8| 57.0 | 80.5 | 9.4 | 49.0 | 22.5 |72.6 | 73.8 | 69.9| 7.2 | 15.0| 35.3 | 54.7 | 75.0 | 29.4 | 46.9|

## VOC 12
|impl| aeroplane | bicycle |  bird |  boat | bottle |  bus  |  car  |  cat  | chair |  cow  | diningtable |  dog  | horse | motorbike | person | pottedplant | sheep |  sofa | train | tvmonitor |  Mean |
|-----------|---------|-------|-------|--------|-------|-------|-------|-------|-------|------|------|------|------|------|------|------|------|------|------|------|----|
|here(DDT)|  74.8   |   67.4  | 60.3 | 21.6 |  28.3  | 81.6 | 64.0 | 81.8 |  31.9 | 55.8 |     33.3    | 75.4 |  72.4 |    71.0   |  37.7  |     27.0    |  44.1 | 51.4 |  43.2 |    24.2   | 52.4 |
|paper | 76.7 | 67.1 | 57.9|30.5| 13.0 | 81.9| 48.3 | 75.7 | 18.4 | 48.8 | 27.5 |71.8 | 66.8 | 73.5| 6.1 | 18.5| 38.0 | 54.7 | 78.6 | 34.6 | 49.4|

