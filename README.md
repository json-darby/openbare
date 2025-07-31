# Openbare: Landmark-Guided Inpainting for Occluded Face Recognition

<!--
<pre style="width: 100%; overflow-x: auto;">
████▄ █ ▄▄  ▄███▄      ▄   ███   ██   █▄▄▄▄ ▄███▄   
█   █ █   █ █▀   ▀      █  █  █  █ █  █  ▄▀ █▀   ▀  
█   █ █▀▀▀  ██▄▄    ██   █ █ ▀ ▄ █▄▄█ █▀▀▌  ██▄▄    
▀████ █     █▄   ▄▀ █ █  █ █  ▄▀ █  █ █  █  █▄   ▄▀ 
       █    ▀███▀   █  █ █ ███      █   █   ▀███▀   
        ▀           █   ██         █   ▀            
                                  ▀                                                                                                                                               
</pre>
-->

Openbare tackles the challenging problem of face verification when the lower half of a person's face is obscured, such as by a medical mask. 
This project is a straightforward and powerful approach that uses advanced inpainting technology, guided by facial landmarks, to reconstruct a person's features. 
This would not only make identification easier, but also enhance operational efficiency and patient safety by reducing the need for unnecessary mask removals, 
which in turn reduces the exposure to certain pathogens.

## System Workflow
This diagram illustrates the flow of information and interaction between the system's core components, from user login to secure output. It acts as a visual guide to the project's architecture.

<p align="center">
  <img src="https://github.com/user-attachments/assets/024dd6c6-b2e5-4f11-8280-9c62999f4da0" alt="Openbare System Architecture Flowchart" width="75%" />
</p>

Once a user logs in, their credentials are verified against a secure SQLite database. They can then upload images or capture them via webcam. 
The system automatically detects masks using either a cloud-based Roboflow or local Detectron2 model. The image is then passed through one of three inpainting modes: a basic blind approach, 
a mapping-based one, or a hybrid version. The final results can be saved, at which point they are encrypted with AES-256 and an audit trail is created. 
Unsaved images are deleted when the session ends to ensure data safety.

## A Visual Showcase

### The Application in Action
<img src="https://github.com/user-attachments/assets/ba27640a-d2f9-4600-996b-1dd6c0952dfc" width="100%" />
This is what the application looks like. The user has selected a masked image on the right, and the application has generated a reconstruction on the left. The user can then choose to save the output.

### Landmark Guidance
<img src="https://github.com/user-attachments/assets/180dbc8d-8cf0-4e9a-813c-f5c6be85983a" width="100%"/>
This shows how facial landmarks are used to guide the inpainting model (mapped and hybrid), ensuring that the regenerated features are structurally accurate.

---
## Contact & Demo

For questions or assistance, please feel free to reach out:

**Email:** jsndarby1@gmail.com

You can also view a full demonstration of the Openbare project's functionality in this YouTube video:

**Project Demo:** [Openbare](https://youtu.be/GhLgbmP5vSE)
