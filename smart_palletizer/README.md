# smart_palletizer

[[_TOC_]]

### 1. 2D boxes detection

---

The goal of this task is to detect and small box, and medium box from color/depth images:

Note that you are free to use classical detection methods, or even [**synthetic**](https://github.com/DLR-RM/BlenderProc) data generated using the provided mesh files to achieve this task.

Here is an example of detected medium box:
![medium_box](./docs/imgs/medium_box.png)


### 2. Planar patches detection (3D)

---

The goal of this task is to detect planar surfaces in the point cloud of the boxes that might represent any of box sides and group them according to the box that they belong to.

![planar_patches](./docs/imgs/planar_patches.png)

### 3. Point Cloud post processing

---

Raw Point Clouds provided in the data folder are noisy, the goal of this task is to post-process the pointcloud to get a clean pointcloud for further processing, without jeopardizing the dimensions of the box too much.

![clean_cloud](./docs/imgs/clean_cloud.png)

### 4. Boxe Poses Estimation

---

This task aims to estimate 6D poses (Translation, Orientation) of the boxes in the scene:

![boxes_poses](./docs/imgs/boxes_poses.png)

## Evaluation

1. **Methodology** correctness into solving the challenge, please explain your efforts into solving the challenge rather than sending code only.
1. **Code validity** your code for the submitted tasks has to compile on our machines, hence we ask you kindly to provide clear instructions on how to compile/run your code, please don't forget to mention depndency packages with their versions to reproduce your steps.
3. **Code Quality** we provide empty templates e.g. `.gitignore`, `docker`, `CI`, Documentation, they are **optional**, keep in mind that best practices are appreciated and can add **extra points** to your submission.
4. **Visualization** it would be nice if you can provide visual results of what you have done: images, videos, statistics to represent your results.
5. **ChatGPT / Gemini** are useful tools if you use them wisely, however original work / ideas are always regarded with higher appreciation and gain more points, we remind you that we might fail the challenge if you misuse them (*e.g. copy paste code without understanding*).

## Documentation

Documenting your code is appreciated as you can explain the functionality in standardized way, hence we provide you with a Python template to compile your [documented functions/classes](https://www.geeksforgeeks.org/python-docstrings):

```sh
sudo apt-get install texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra latexmk
#
cd smart_palletizer/docs
pip3 install -U pip
pip3 install -r requirements.txt
make clean && sphinx-apidoc -f -o source ../src/smart_palletizer
make html SPHINXBUILD="python3 <path_to_sphinx>/sphinx-build"
##----------
## Example:
##----------
# make html SPHINXBUILD="python3 $HOME/venv/bin/sphinx-build"
```
---

If you are using C++, then please refer to [Doxygen](https://www.doxygen.nl)
