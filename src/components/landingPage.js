import React, { useState, useEffect } from 'react';
import './landingPage.css';
import Slide from 'react-reveal/Slide';
import GridLoader from 'react-spinners/GridLoader';

function ImageUpload() {
  const [file, setFile] = useState(null);
  const [image, setImage] = useState({ src: null, name: null });
  const [predictionMask, setPredictionMask] = useState(null);
  const [overlayedImage, setOverlayedImage] = useState(null);
  const [showLoader, setShowLoader] = useState(false);
  const [showSlide, setShowSlide] = useState(false);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);

    const reader = new FileReader();
    reader.onload = () => {
      setImage({ src: reader.result, name: selectedFile.name });

    };

    reader.readAsDataURL(selectedFile);
   
  };

  useEffect(() => {
    if (showSlide && predictionMask) {
      handleOverlayedImage();
    }
  }, [showSlide, predictionMask]);

  const handleSubmit = async (event) => {
    event.preventDefault();

    const formData = new FormData();
    formData.append('image', file);
    setShowLoader(true);
    try {
      const response = await fetch('/get_prediction_mask', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setPredictionMask(data.pred_mask.slice(2, -1));
      console.log(data.pred_mask.slice(2, -1));
      
      setTimeout(() => {
        setShowLoader(false);
        setShowSlide(true);
        handleOverlayedImage();
      }, 3000);
    } catch (error) {
      console.error('There was a problem with the fetch operation:', error);
    }

  
  };



  const handleOverlayedImage = async () => {
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('/get_overlayed_image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data1 = await response.json();
      setOverlayedImage(data1.overlayed_image.slice(2, -1));
    } catch (error) {
      console.error('There was a problem with the fetch operation:', error);
    }
  };

  return (
    <div>
      <h2>Water Body Detection</h2>
      <form onSubmit={handleSubmit}>
      <label for="file-upload" class="button">Upload</label>
        <input id="file-upload" className='button displayNone' type="file" accept="image/*" onChange={handleFileChange} />
        <button className='button1' type="submit">Submit</button>
      </form>

      {file && (
        <div className="image-container">
          <div>
            <img src={image.src} alt="Input Image" />
            <p>Input Image</p>
          </div>
          {showLoader ? (
            <div className="loader-container">
              <GridLoader color="#36d7b7" />
            </div>
          ) : (
            showSlide && predictionMask && (
              <Slide left cascade>
                <div>
                  <img src={`data:image/jpeg;base64,${predictionMask}`} alt="Output mask" />
                  <p>Output mask</p>
                </div>
              </Slide>
            )
          )}
          {overlayedImage && (
            <Slide left cascade>
              <div>
                <img src={`data:image/jpeg;base64,${overlayedImage}`} alt="Overlayed Image" />
                <p>Overlayed Image</p>
              </div>
            </Slide>
          )}
        </div>
      )}
    </div>
  );
}

export default ImageUpload;

