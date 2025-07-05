# Video Annotation Viewer - Multimodal Analysis Tool

A sophisticated web-based video annotation viewer designed for reviewing offline-processed videos with rich multimodal annotations including pose detection, facial emotion recognition, audio analysis, and synchronized timeline visualization.

## Features

### 🎥 **Video Player with Multimodal Overlays**
- **Pose Detection**: Real-time skeleton overlay with keypoints and connections
- **Facial Emotion Recognition**: Bounding boxes with emotion labels and confidence scores
- **Audio Emotion Analysis**: Voice sentiment indicators synchronized with speech
- **Subtitles**: Transcribed speech with customizable styling
- **Event Annotations**: Action and interaction labels with timing

### 📊 **Rich Timeline Visualization**
- **Audio Waveform**: Visual representation of audio amplitude over time
- **Motion Intensity**: Graph showing movement activity throughout the video
- **Event Tracks**: Subtitle and action event segments with hover details
- **Interactive Navigation**: Click and drag to seek, with precise time markers

### 🎛️ **Advanced Controls**
- **Overlay Toggles**: Show/hide individual annotation layers
- **Playback Controls**: Play, pause, frame stepping, and variable speed playback
- **Timeline Settings**: Customize which tracks are visible
- **Responsive Design**: Optimized for research and analysis workflows

## How to Use

### 1. **Getting Started**
- Launch the application to see the welcome screen
- Click "Get Started" to begin loading your files
- Or click "Try with Sample Data" to explore with demo content

### 2. **Loading Your Data**
- **Video File**: Upload any standard video format (MP4, WebM, AVI, MOV)
- **Annotation File**: Upload a JSON file containing your analysis results
- Click "Start Viewing" to launch the annotation viewer

### 3. **JSON Data Format**
Your annotation JSON should include:
```json
{
  "video": {
    "filename": "your-video.mp4",
    "duration": 60.0,
    "frameRate": 30,
    "width": 1920,
    "height": 1080
  },
  "frames": {
    "0": {
      "frameNumber": 0,
      "timestamp": 0.0,
      "persons": [...],
      "faces": [...],
      "audioEmotion": {...},
      "motionIntensity": 0.3
    }
  },
  "events": [...],
  "audio": {...},
  "metadata": {...}
}
```

### 4. **Navigation & Analysis**
- **Timeline**: Click anywhere to jump to that time
- **Overlays**: Toggle different annotation types using the control panel
- **Playback**: Use standard controls plus frame stepping for detailed analysis
- **Timeline Tracks**: Enable/disable waveform, motion, and event visualization

## Technology Stack

- **Frontend**: React + TypeScript + Tailwind CSS
- **Video Processing**: HTML5 Video API with Canvas overlays
- **Timeline**: Custom SVG/Canvas implementation with real-time synchronization
- **UI Components**: Shadcn/ui for consistent, accessible interface
- **Build System**: Vite for fast development and optimized builds

## Use Cases

### 🔬 **Research Applications**
- Human behavior analysis and validation
- Computer vision pipeline result review
- Multimodal interaction studies
- Algorithm performance evaluation

### 📹 **Video Analysis**
- Pose estimation result verification
- Emotion recognition quality assessment
- Audio-visual synchronization review
- Event detection validation

### 🏥 **Clinical & Educational**
- Therapy session analysis
- Movement disorder assessment
- Educational interaction studies
- Social behavior research

## Getting Started for Developers

This project was built using [Lovable](https://lovable.dev), an AI-powered web development platform.

### Running Locally
```bash
npm install
npm run dev
```

### Customization
The application is designed to be easily extensible:
- Add new annotation types by extending the overlay system
- Customize timeline tracks for your specific data
- Modify the JSON schema to match your pipeline outputs
- Integrate with your existing analysis workflow

## Future Enhancements

- **Editing Capabilities**: Validate and correct annotations directly in the viewer
- **Multi-video Comparison**: Side-by-side analysis of different processing results
- **Export Features**: Generate reports and export corrected annotations
- **Real-time Processing**: Integration with live analysis pipelines

## Support

For questions, feature requests, or technical support, please refer to the project documentation or contact the development team.

---

**Built with ❤️ using Lovable** - *AI-powered full-stack development*
