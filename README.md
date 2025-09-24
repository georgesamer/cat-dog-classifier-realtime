# cat-dog-classifier-realtime
Real-time Cat vs Dog classifier using OpenCV, MediaPipe, and a pre-trained MobileNetV2 model.  The app accesses your webcam, detects faces, and classifies frames as Cat, Dog, or Not Cat/Dog  with live feedback on the video stream.
name: Cat/Dog Classifier CI/CD

# Trigger the workflow on push and pull requests to main branch
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

# Define environment variables
env:
  PYTHON_VERSION: '3.9'
  TF_CPP_MIN_LOG_LEVEL: '2'  # Reduce TensorFlow logging

jobs:
  # Job 1: Code Quality and Linting
  code-quality:
    runs-on: ubuntu-latest
    name: Code Quality Check
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install linting dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 black isort pylint
        
    - name: Run Black (Code Formatting Check)
      run: |
        black --check --diff .
        
    - name: Run isort (Import Sorting Check)
      run: |
        isort --check-only --diff .
        
    - name: Run Flake8 (Style Guide Enforcement)
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

  # Job 2: Testing
  test:
    runs-on: ubuntu-latest
    needs: code-quality
    name: Run Tests
    
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
        
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
        
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
          
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov pytest-mock
        pip install opencv-python-headless  # Headless version for CI
        pip install tensorflow
        pip install mediapipe
        pip install numpy
        
    - name: Create test file
      run: |
        mkdir -p tests
        cat > tests/test_classifier.py << 'EOF'
        import pytest
        import numpy as np
        import cv2
        from unittest.mock import Mock, patch
        import sys
        import os
        
        # Add the parent directory to sys.path to import our module
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        class TestCatDogClassifier:
            @patch('tensorflow.keras.applications.MobileNetV2')
            @patch('mediapipe.solutions.face_detection.FaceDetection')
            def test_classifier_initialization(self, mock_face_detection, mock_mobilenet):
                """Test if classifier initializes without errors"""
                from cat_dog_classifier import CatDogClassifier
                
                # Mock the model loading
                mock_model = Mock()
                mock_mobilenet.return_value = mock_model
                
                classifier = CatDogClassifier()
                assert classifier is not None
                
            def test_preprocess_image(self):
                """Test image preprocessing"""
                from cat_dog_classifier import CatDogClassifier
                
                with patch('tensorflow.keras.applications.MobileNetV2'):
                    with patch('mediapipe.solutions.face_detection.FaceDetection'):
                        classifier = CatDogClassifier()
                        
                        # Create a dummy image
                        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
                        
                        # Test preprocessing
                        preprocessed = classifier.preprocess_image(dummy_image)
                        
                        # Check output shape (batch_size=1, height=224, width=224, channels=3)
                        assert preprocessed.shape == (1, 224, 224, 3)
                        
            def test_cat_dog_classes_defined(self):
                """Test if cat and dog classes are properly defined"""
                from cat_dog_classifier import CatDogClassifier
                
                with patch('tensorflow.keras.applications.MobileNetV2'):
                    with patch('mediapipe.solutions.face_detection.FaceDetection'):
                        classifier = CatDogClassifier()
                        
                        # Check if classes are defined and not empty
                        assert len(classifier.cat_classes) > 0
                        assert len(classifier.dog_classes) > 0
                        
                        # Check some common breeds exist
                        assert 'golden_retriever' in classifier.dog_classes
                        assert 'Persian_cat' in classifier.cat_classes
        EOF
        
    - name: Run tests
      run: |
        pytest tests/ -v --cov=. --cov-report=xml
        
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  # Job 3: Security Scan
  security:
    runs-on: ubuntu-latest
    name: Security Scan
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Run Bandit Security Scan
      uses: securecodewarrior/github-action-bandit@v1.0.1
      with:
        path: "."
        exit_zero: true
        
    - name: Run Safety Check (Dependency Vulnerabilities)
      run: |
        pip install safety
        safety check --json --output safety-report.json || true
        
    - name: Upload safety report
      uses: actions/upload-artifact@v3
      with:
        name: safety-report
        path: safety-report.json

  # Job 4: Build and Package
  build:
    runs-on: ubuntu-latest
    needs: [code-quality, test]
    name: Build Package
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build setuptools wheel
        
    - name: Create setup.py
      run: |
        cat > setup.py << 'EOF'
        from setuptools import setup, find_packages
        
        setup(
            name="cat-dog-classifier",
            version="1.0.0",
            description="Real-time cat/dog classification using webcam",
            author="Your Name",
            author_email="your.email@example.com",
            packages=find_packages(),
            install_requires=[
                "opencv-python>=4.5.0",
                "tensorflow>=2.8.0",
                "mediapipe>=0.8.0",
                "numpy>=1.21.0",
            ],
            python_requires=">=3.8",
            classifiers=[
                "Development Status :: 4 - Beta",
                "Intended Audience :: Developers",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10",
                "Programming Language :: Python :: 3.11",
            ],
        )
        EOF
        
    - name: Build package
      run: |
        python -m build
        
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: python-package-distributions
        path: dist/

  # Job 5: Create Release (only on main branch with tags)
  release:
    runs-on: ubuntu-latest
    needs: [build]
    name: Create Release
    if: startsWith(github.ref, 'refs/tags/v')
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Download build artifacts
      uses: actions/download-artifact@v3
      with:
        name: python-package-distributions
        path: dist/
        
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        body: |
          ## Changes in this Release
          - Real-time cat/dog classification
          - MediaPipe face detection
          - Pre-trained MobileNetV2 model
          - OpenCV webcam integration
          
          ## Installation
          ```bash
          pip install opencv-python tensorflow mediapipe numpy
          ```
          
          ## Usage
          ```bash
          python cat_dog_classifier.py
          ```
        draft: false
        prerelease: false
        
    - name: Upload Release Assets
      uses: actions/upload-release-asset@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        upload_url: ${{ steps.create_release.outputs.upload_url }}
        asset_path: dist/
        asset_name: cat-dog-classifier-dist
        asset_content_type: application/zip

  # Job 6: Performance Test (Optional)
  performance:
    runs-on: ubuntu-latest
    needs: test
    name: Performance Test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install opencv-python-headless tensorflow mediapipe numpy pytest-benchmark
        
    - name: Run performance tests
      run: |
        mkdir -p tests
        cat > tests/test_performance.py << 'EOF'
        import pytest
        import numpy as np
        import time
        from unittest.mock import Mock, patch
        import sys
        import os
        
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        @patch('tensorflow.keras.applications.MobileNetV2')
        @patch('mediapipe.solutions.face_detection.FaceDetection')
        def test_classification_performance(mock_face_detection, mock_mobilenet, benchmark):
            """Test classification performance"""
            from cat_dog_classifier import CatDogClassifier
            
            # Mock the model
            mock_model = Mock()
            mock_model.predict.return_value = np.random.random((1, 1000))
            mock_mobilenet.return_value = mock_model
            
            classifier = CatDogClassifier()
            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Benchmark the preprocessing step
            result = benchmark(classifier.preprocess_image, dummy_image)
            assert result is not None
        EOF
        
        pytest tests/test_performance.py --benchmark-only --benchmark-json=benchmark.json
        
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmark.json
