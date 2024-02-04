### Deep Learning School, part 1, final project.
# Animal detection for state parks

_Structure from the problem definition document is preserved._

## Problem search and solution description

### Problem Identification

When I visited a state park, I saw many different animals, but I couldn't distinguish the species. I thought that it would be great if I could take a photo of an animal, upload it to a state park's application or website and get information about the animal I saw.

### Problem Formulation

When a person visits state park and sees an animal it can be difficult to understand what it is. And as parks are big enough, it's unlikely that there is a staff member or a poster with the exact animal close to you. The visitor needs something which will answer who he sees and provide more information about the animal.

### Target Audience

Visitors of state parks who want to know what animals they see and to learn more about them.

### Solution

I want to create a web-application which will provide the following UX.

As a state park visitor:
- open a website and choose a state park
- upload a photo of an animal or animals you don't know
- get back your photo with a caption for each animal on it
- get the list of animals you saw and links to read more information about each one
- links with information can be opened by clicking on them in the list below the photo or by clicking on the animal on the photo
- if you need, you can return back to the state park choosing

As a state park representative:
- open a website and choose to add a new state park
- register and prove that you are the park representative
- set a list of animals in your state park
- if there are animals, which are new for the model, app suggests to provide some photoes of them and label them to train the model
- add information about the animals to create pages about them or provide links to your own website with information
- add the state park to the app
- get a working state park page in the app to share with visitors

In this project I will focus on the visitor experience and provide an MVP wich allows to do all the listed visitor's actions.

## Search for a pre-trained model and dataset

I use mmdetection framework and an rtmdet-tiny model from it. This model has been trained on COCO dataset, which contains 80 classes. 10 of the classes are animals, so for demonstration purposes I will use images with these animals from COCO and google. In futher development the model can be trained additionally on a dataset with animals from the particular state park. I processed results of the model to show only animals. Here are some examples of final results.

![horse_and_dog](https://github.com/seleznevajane/dls1_final_project/assets/81403952/5e3dfd85-07d3-4a7d-9486-60d5814228f6)
![cows_and_sheeps](https://github.com/seleznevajane/dls1_final_project/assets/81403952/a29b8745-b91c-4223-9d88-be93669f1e1b)
![bird](https://github.com/seleznevajane/dls1_final_project/assets/81403952/bf60eaba-23ad-4f91-af14-db13acb822a9)
![bear](https://github.com/seleznevajane/dls1_final_project/assets/81403952/4ea3010f-fe51-4d1c-ae33-3f2e5b36fdbe)
![zebra_and_elephant_and_giraffe](https://github.com/seleznevajane/dls1_final_project/assets/81403952/d64d0da2-1a80-4276-a9dc-7884bc1a3a72)
![zebra_and_elephant](https://github.com/seleznevajane/dls1_final_project/assets/81403952/9e192f3f-67fc-421e-9c64-b214a87c8884)

## Demo development

I use FastAPI. The application has 2 demo pages and 3 handlers. Start page contains a list of links to pages of differenet state parks, in demo there is only one "Example Park". By clicking on "Example Park" link we go to Park's page. This page contains an upload form for an image, which is processed by the corresponding post handler. The handler returns result image and some text data for html to show. And a user sees his image with bounded and labeled animals which are clickable and a text which contains a list of animals on the photo and links to their information pages. In demo this pages are taken from Wikipedia.

Handlers are described in the [main.py](code/main.py).

## Adding detector model to the demo

I use DetInferencer from mmdetection and then process its results to leave bounding boxes only for animals and prepare a list of nonrepeating animals for the text.

Detection and postprocessing logic is described in the [processing.py](code/processing.py)

## Demo testing

### Demo Interface

Here are some images showing the interface described above.
<img width="1080" alt="start_page" src="https://github.com/seleznevajane/dls1_final_project/assets/81403952/808d50d1-4621-4e96-ba5f-43082ebd5367">
<img width="1080" alt="example_park_page" src="https://github.com/seleznevajane/dls1_final_project/assets/81403952/3fb0ba9e-37b1-42a3-96d8-d091b8a0bdf9">
<img width="1080" alt="example_park_page_no_animals" src="https://github.com/seleznevajane/dls1_final_project/assets/81403952/d24fe136-a652-4a08-ae50-fce4708c8245">
<img width="1080" alt="example_park_page_results" src="https://github.com/seleznevajane/dls1_final_project/assets/81403952/56307fe6-9098-4458-9072-4995bebb2bcb">

### Model Inference

As was seen above, the model works well on images, where animals are not covered with anything and are not too close to each other. But when animals are too close the model can miss some of them as on following pictures.
![cats_and_dog](https://github.com/seleznevajane/dls1_final_project/assets/81403952/83483246-7974-4aa2-a4be-0ffd3b9f7bf8)
![cats](https://github.com/seleznevajane/dls1_final_project/assets/81403952/cd801e77-d4ff-4997-a0e8-971f5fd40275)
Also if an animal is obscured by branches the model may not notice it or consider it as several animals as on following pictures.
![bird_2](https://github.com/seleznevajane/dls1_final_project/assets/81403952/fe87fd6c-b17c-48fc-b484-8dc3654878a5)
![bird_1](https://github.com/seleznevajane/dls1_final_project/assets/81403952/38e67c83-f228-4226-aebc-d50c8d7c08bc)
I think that the model can be trained further on difficult images to be able to deal with them.

## Design improvement or application deployment

For design purposes I made bounding boxes clickable. During inference result processing I prepare a list of boxes coordinates and corresponding links. In [example_park.html](/code/templates/example_park.html) I add this links with a map.

For the application deployment I created a dockerfile to build a container and tested it.
