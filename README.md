# Maternalyze - Pregnancy ML Prediction Portal

<p>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  </a>
  <a href="https://python.org/">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://lightgbm.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/LightGBM-00C853?style=for-the-badge&logo=lightgbm&logoColor=white" alt="LightGBM">
  </a>
  <a href="https://pandas.pydata.org/">
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  </a>
  <a href="https://numpy.org/">
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  </a>
  <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript">
    <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript">
  </a>
  <a href="https://www.chartjs.org/">
    <img src="https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white" alt="Chart.js">
  </a>
</p>

## Overview

Maternalyze is a web-based Machine Learning prediction portal focusing on pregnancy-related health conditions including Gestational Diabetes Mellitus (GDM) and Child Outcome predictions. The portal collects user input, sends it to ML models via backend APIs, and displays beautified, user-friendly prediction results with charts.

## Features

- Structured input forms with clear labels and unit helpers for each parameter.
- Beautified display of prediction results as tables and styled lists.
- Interactive charts showing prediction probabilities and feature importance.
- Smooth navigation between GDM and Child Outcome prediction tools.
- User interface styled with CSS for a clean and consistent experience.

## Installation

1. Clone the repository.  
2. Ensure you have a backend API running and accessible — configure the API base URL in `config.js`.  
3. Open the HTML files (`index.html`, `gdm_prediction.html`, `child_prediction.html`) in a modern web browser.  
4. The frontend sends requests to the specified backend API for predictions.

## Usage

- Use the main portal (`index.html`) to navigate between prediction pages.
- Fill in required input fields with appropriate values and units.
- Submit the form to see detailed, formatted prediction outputs.
- Review visual probability charts and feature importance graphs.
- Use the "Back to Home" button to return to the main portal.

## File Structure

- `index.html` — Landing page with navigation and overview chart.  
- `gdm_prediction.html` — Gestational Diabetes prediction form and results.  
- `child_prediction.html` — Child Outcome prediction form and results.  
- `style.css` — Stylesheet with layout and component styling.  
- `script.js` — JavaScript handling form submission, JSON parsing, and chart rendering.  
- `config.js` — Configuration with backend API URL.