# MRIxFields2026 Task 3 Docker

This folder is a challenge-test Docker wrapper around the conditioned Swin-UNETR in the supplied project.

## 1. Add the trained checkpoint

Copy the final trained checkpoint into:

    app/weights/model.pth

The source project ZIP did not contain a .pth/.pt checkpoint, so the Docker image is not complete until this file is added.

## 2. Build

From this folder:

    docker build -t mrixfields-task3:v1 .

## 3. Local challenge-style test

Prepare a local test input folder whose structure follows the challenge test guide:

    test_input/
      manifest.json
      T1W/.../src/*.nii.gz

Then run:

    mkdir -p test_output
    docker run --rm --gpus all \
      -v "$PWD/test_input:/input:ro" \
      -v "$PWD/test_output:/output" \
      mrixfields-task3:v1

The container must terminate with a line like:

    SUCCESS: wrote exactly N predictions.

## 4. Tag for Synapse

Replace YOUR_PROJECT_ID with the Synapse project that owns your Docker repository:

    docker tag mrixfields-task3:v1 docker.synapse.org/YOUR_PROJECT_ID/task3:v1

Example format used by MRIxFields2026:

    docker.synapse.org/syn75078640/task3:v1

## 5. Log in and push

    docker login docker.synapse.org
    docker push docker.synapse.org/YOUR_PROJECT_ID/task3:v1

Use a NEW tag for each resubmission, e.g. v2, v3, ...

## Challenge interface implemented here

- Reads `/input/manifest.json`.
- Uses each manifest sample's exact `input`, `output`, `source_field`, `target_field`, and `modality`.
- Writes only `.nii.gz` predictions under `/output`.
- Preserves full input volume shape and orientation.
- Clips predictions to [0,1] and rejects NaN/Inf.
- Produces no segmentation files.
