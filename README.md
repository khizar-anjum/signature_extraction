Hello, this repo is for signature extraction by the usage of neural 
networks.  
The main objective of this project is to provide a function that takes 
in pdf files and extracts all the signatures as well the text in them 
using neural networks and OCR techniques. 

# Getting Started
To get started, install all the dependencies via Python pip manager.  
`pip install -r requirements.txt`  
This will take care of the packages required for proper working of the `extractor` class except the installation of PyTorch. To get PyTorch installed, I suggest you go to [PyTorch](https://pytorch.org) homepage and get it installed according to your system's specifications.  
  
If you have GPU capability on your system, do not forget to avail it by properly editing `extractor.__load_model()` method in `extractor.py` file i.e. by setting `device = torch.device('gpu')`.  
  
# Usage
First download the pre-trained Siamese Convolutional Neural Network by running `python DownloadModel.py`.  
The pretrained weights and code for Siamese CNN's have been taken from [OfflineSignatureVerification](https://github.com/Aftaab99/OfflineSignatureVerification) repo.  
  
Afterwards, use the `extractor` class to perform all the necessary functions. Proper documentation has been provided for all the functions. A simple use case has also been  provided in `main.py` file.


# References
[1] Thanks to Aftaab99 for amazing work on Siamese Neural Networks for [Offline Signature Verfication](https://github.com/Aftaab99/OfflineSignatureVerification).  
[2] Thanks to rbaguilla for their [repo](https://github.com/rbaguila/document-layout-analysis) stipulating the detection of lines, words and paragraphs in a scanned docuemnt. 

