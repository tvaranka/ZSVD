<p align="center">

  <h2 align="center">ZSVD: Zero-Shot Video Deraining with Video Diffusion Models</h2>
  <p align="center">
                <a href="https://scholar.google.com/citations?user=5QWyHT4AAAAJ&hl=en">Tuomas Varanka</a>,
                <a href="https://sites.google.com/view/juan-luis-gb/home">Juan Luis Gonzalez</a>,
                <a href="https://scholar.google.com/citations?user=lDekV6IAAAAJ&amp;hl=en&amp;oi=ao">Hyeongwoo Kim</a>,
                <a href="https://www.linkedin.com/in/pablo-garrido-485472169/">Pablo Garrido</a>,
                <a href="https://xu-yao.github.io/">Xu Yao</a>
    <br>
    <br>
    <b>&nbsp;  <img src="./assets/logo.png" alt="Flawless AI Logo" style="height:1.2em; vertical-align:middle; margin-right:0.2em;"> Flawless AI </b>
    <br>
    <br>
        <a href=""><img src='https://img.shields.io/badge/arXiv-ZSVD-red' alt='Paper PDF'></a>
        <a href=""><img src='https://img.shields.io/badge/Project_Page-ZSVD-green' alt='Project Page'></a>
    <br>
  </p>
  
  <table align="center">
    <tr>
    <video autoplay muted loop playsinline height="100%">
            <source src="./assets/teaser.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </tr>
  </table>

This repository contains the deraining code of the WACV 2026 paper [ZSVD](https://arxiv.org/list/cs.CV/recent). 

## Installation
We recommend to use uv for fast installation.
```shell
# Create & activate environment
uv venv
source .venv/bin/activate
# Install dependencies
uv pip install -r requirements.txt
```

## Testing

```shell
bash run_example.sh
```

### parameters:
```shell
python src/inference_path.py \
    --input_path examples \
    --negative_prompt "heavy rain, light rain, drizzle, sprinkle, shower, raindrops, rainstorm, streak" \
    --skip_value 40 \
    --editing_guidance_scale 15
```

`--input_path`, a folder with `.mp4` videos in it.

`--skip_value`, value from which the deraining process is started. Default inference steps is 100, therefore `skip_value` corresponds to starting after 40% of the inversion. Recommend value between `10-50`, higher values retain more information from the original scene, while lower values enable more robust deraining, but at the cost of losing structural information.

`--editing_guidance_scale`, CFG value used for deraining. Larger value corresponds to improved deraining effect, but at the cost of artifacts and losing structural information. Recommended value between `10-20`.

`--negative_prompt`, the set of negative prompts used for deraining.

`blocks_to_change`, not included in the CLI interface, but can be found from the [src/inference_path.py L66](https://github.com/flwls/research-video_restoration/blob/8e3bc734d37db7393ba9fcb34c288e3ec2a47e4f/src/inference_path.py#L66). Determines at which blocks the keys and values are changed. 
```python
blocks_to_change = np.array(list(range(4)) + list(range(15, 30)))  # KV swap at beginning (0-4) and end (15-30)
blocks_to_change = np.array(list(range(4)))  # KV swap at the beginning (0-4)
blocks_to_change = np.array([])  # No KV swap
```




## RealRain13 dataset
Here is the collected real rain video dataset [RealRain13](./realrain13/).


## Citation

If you find our work helpful, please cite us.
#TODO: update citation

