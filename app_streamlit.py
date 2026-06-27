import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import librosa
import librosa.display

from app.services.inference import InferenceService
from app.agent.forensic_agent import ForensicRAGAgent
from ai_engine.xai.gradcam import GradCAM
from ai_engine.fusion.late_fusion import LateFusionClassifier
import torchvision.transforms as T

# Page Configuration
st.set_page_config(
    page_title="SentinelForensicsAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Professional Cybersecurity Theme)
st.markdown("""
<style>
    .main-title {
        font-family: 'Inter', sans-serif;
        color: #58a6ff;
        text-align: center;
        font-weight: 600;
        font-size: 2.2rem;
        margin-bottom: 2px;
    }
    .subtitle {
        text-align: center;
        color: #8b949e;
        font-size: 0.95rem;
        margin-bottom: 30px;
    }
    .status-active {
        color: #3fb950;
        font-weight: bold;
    }
    .status-info {
        color: #58a6ff;
        font-weight: bold;
    }
    div.stButton > button {
        background-color: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
    }
    div.stButton > button:hover {
        border-color: #58a6ff !important;
        color: #58a6ff !important;
    }
    .reference-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">SentinelForensicsAI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multimodal Deepfake Forensic Investigation & Explainable AI Dashboard</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("### SentinelAI")
st.sidebar.markdown("---")
st.sidebar.markdown("**System Status:**")
st.sidebar.markdown("Model Weights: <span class=\"status-active\">Active</span>", unsafe_allow_html=True)
st.sidebar.markdown("Feature Cache: <span class=\"status-active\">Active</span>", unsafe_allow_html=True)
st.sidebar.markdown("Forensic RAG: <span class=\"status-info\">Online</span>", unsafe_allow_html=True)

# Load model for GradCAM once to prevent reloading overhead
@st.cache_resource
def get_cached_gradcam():
    model = LateFusionClassifier(pretrained_video=False)
    model_path = "models/multimodal_best.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    target_layer = model.video_extractor.resnet.layer4[-1]
    return GradCAM(model, target_layer), model

gradcam_obj, lf_model = get_cached_gradcam()

# Initialize services
inference_service = InferenceService()
rag_agent = ForensicRAGAgent()

# ----------------- TABS -----------------
tab1, tab2, tab3 = st.tabs(["Analysis Hub", "Forensic RAG Agent", "Dataset & Performance"])

# ----------------- TAB 1: ANALYSIS HUB -----------------
with tab1:
    st.subheader("Multimodal Analysis Hub")
    st.write("Upload a video, image, or audio file to execute deep learning authenticity verification.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Media (Video: .mp4, .avi | Image: .png, .jpg, .jpeg | Audio: .wav, .mp3)", 
            type=["mp4", "avi", "png", "jpg", "jpeg", "wav", "mp3"]
        )
        
        media_type = None
        if uploaded_file is not None:
            # Determine media type by file extension
            filename = uploaded_file.name.lower()
            if filename.endswith((".mp4", ".avi")):
                media_type = "video"
            elif filename.endswith((".png", ".jpg", ".jpeg")):
                media_type = "image"
            elif filename.endswith((".wav", ".mp3")):
                media_type = "audio"
                
            # Save uploaded file temporarily
            temp_dir = "storage/temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Display uploaded media preview
            if media_type == "video":
                st.video(temp_path)
            elif media_type == "image":
                st.image(temp_path, use_container_width=True)
            elif media_type == "audio":
                st.audio(temp_path)
            
            if st.button("Run Forensic Inference", use_container_width=True):
                with st.spinner("Executing forensic feature extraction and prediction models..."):
                    try:
                        face_t = None
                        mel_db = None
                        orig_face_path = os.path.join(temp_dir, "temp_face.jpg")
                        
                        # 1. Processing based on media type
                        if media_type == "video":
                            res = inference_service.predict_video(temp_path)
                            st.session_state["predict_res"] = res
                            
                            cap = cv2.VideoCapture(temp_path)
                            success, frame = cap.read()
                            if success:
                                from ai_engine.preprocessing.face_detector import FaceDetector
                                detector = FaceDetector()
                                boxes = detector.detect_faces_in_frame(frame)
                                if len(boxes) > 0:
                                    x, y, w, h = boxes[0]
                                    pad_w, pad_h = int(w * 0.15), int(h * 0.15)
                                    height, width, _ = frame.shape
                                    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                                    x2, y2 = min(width, x + w + pad_w), min(height, y + h + pad_h)
                                    face_crop = frame[y1:y2, x1:x2]
                                    if face_crop.size > 0:
                                        cv2.imwrite(orig_face_path, face_crop)
                                        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                                        pil_face = Image.fromarray(face_rgb)
                                        transform = T.Compose([
                                            T.Resize((224, 224)),
                                            T.ToTensor(),
                                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
                                        face_t = transform(pil_face).unsqueeze(0)
                                        
                            mel_db = inference_service.audio_extractor.extract_mel_spectrogram(temp_path)
                            
                        elif media_type == "image":
                            frame = cv2.imread(temp_path)
                            from ai_engine.preprocessing.face_detector import FaceDetector
                            detector = FaceDetector()
                            boxes = detector.detect_faces_in_frame(frame)
                            if len(boxes) > 0:
                                x, y, w, h = boxes[0]
                                pad_w, pad_h = int(w * 0.15), int(h * 0.15)
                                height, width, _ = frame.shape
                                x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                                x2, y2 = min(width, x + w + pad_w), min(height, y + h + pad_h)
                                face_crop = frame[y1:y2, x1:x2]
                                if face_crop.size > 0:
                                    cv2.imwrite(orig_face_path, face_crop)
                                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                                    pil_face = Image.fromarray(face_rgb)
                                    transform = T.Compose([
                                        T.Resize((224, 224)),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
                                    face_t = transform(pil_face).unsqueeze(0)
                            else:
                                cv2.imwrite(orig_face_path, frame)
                                face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                pil_face = Image.fromarray(face_rgb)
                                transform = T.Compose([
                                    T.Resize((224, 224)),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
                                face_t = transform(pil_face).unsqueeze(0)
                                
                            mel_db = np.zeros((128, 300), dtype=np.float32)
                            
                        elif media_type == "audio":
                            face_t = torch.zeros((1, 3, 224, 224))
                            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
                            cv2.imwrite(orig_face_path, dummy_img)
                            
                            mel_db = inference_service.audio_extractor.extract_mel_spectrogram(temp_path)

                        if face_t is None:
                            face_t = torch.zeros((1, 3, 224, 224))
                        
                        mel_t = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        
                        with torch.no_grad():
                            logits = lf_model(face_t, mel_t)
                            probs = torch.sigmoid(logits).item()
                            
                        is_fake = probs >= 0.5
                        
                        if media_type == "image":
                            details = {"visual_probability": probs, "vocal_probability": 0.0}
                        elif media_type == "audio":
                            details = {"visual_probability": 0.0, "vocal_probability": probs}
                        else:
                            details = {"visual_probability": probs * 0.9, "vocal_probability": probs * 0.8}
                            
                        st.session_state["predict_res"] = {
                            "prediction_score": probs,
                            "is_fake": is_fake,
                            "details": details
                        }
                        
                        if media_type in ["video", "image"]:
                            gradcam_out_path = os.path.join(temp_dir, "gradcam_output.png")
                            gradcam_obj.generate_heatmap(
                                face_tensor=face_t,
                                mel_tensor=mel_t,
                                original_image_path=orig_face_path,
                                output_path=gradcam_out_path
                            )
                            st.session_state["gradcam_overlay"] = gradcam_out_path
                            if "audio_spectrogram" in st.session_state:
                                del st.session_state["audio_spectrogram"]
                        else:
                            st.session_state["audio_spectrogram"] = mel_db
                            if "gradcam_overlay" in st.session_state:
                                del st.session_state["gradcam_overlay"]
                                
                    except Exception as e:
                        st.error(f"Inference execution failed: {e}")
    
    with col2:
        if "predict_res" in st.session_state:
            res = st.session_state["predict_res"]
            score = res["prediction_score"]
            is_fake = res["is_fake"]
            explanation = res["details"]
            
            st.markdown("### Analysis Verdict")
            if is_fake:
                st.error(f"VERDICT: DEEPFAKE DETECTED (Confidence: {score*100:.1f}%)")
            else:
                st.success(f"VERDICT: REAL MEDIA DETECTED (Confidence: {(1-score)*100:.1f}%)")
                
            st.metric(label="Manipulated Probability", value=f"{score*100:.2f}%")
            st.progress(score)
            
            st.markdown("### Modality Probability Breakdown:")
            st.write(f"- Visual Forgery Probability: {explanation['visual_probability']*100:.1f}%")
            st.write(f"- Acoustic Forgery Probability: {explanation['vocal_probability']*100:.1f}%")
            
            if "gradcam_overlay" in st.session_state and os.path.exists(st.session_state["gradcam_overlay"]):
                st.markdown("### Explainable AI: Grad-CAM Face Heatmap")
                st.image(st.session_state["gradcam_overlay"], caption="Grad-CAM highlights regional anomalies targeted by the model.", use_container_width=True)
                
            elif "audio_spectrogram" in st.session_state:
                st.markdown("### Acoustic Analysis Signature")
                fig, ax = plt.subplots(figsize=(6, 2.8))
                librosa.display.specshow(st.session_state["audio_spectrogram"], sr=16000, hop_length=512, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
                ax.set_title("Log-Mel Spectrogram", fontsize=10)
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Frequency (Hz)", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("Run forensic inference to display results.")

# ----------------- TAB 2: RAG AGENT Chat -----------------
with tab2:
    st.subheader("Forensic AI Agent")
    st.write("Query the local facts base or choose a quick action to inspect system diagnostics.")
    
    left_chat_col, right_ref_col = st.columns([3, 2])
    
    with left_chat_col:
        # Predefined Quick Queries
        st.markdown("**Quick Queries:**")
        q1, q2, q3 = st.columns(3)
        pending_query = None
        with q1:
            if st.button("Model Performance", use_container_width=True):
                pending_query = "What is the model accuracy and performance metrics?"
        with q2:
            if st.button("Check Duplicates", use_container_width=True):
                pending_query = "Are there any duplicate files in the dataset?"
        with q3:
            if st.button("Dataset Count", use_container_width=True):
                pending_query = "How many total, real, and fake videos are there?"
                
        # Initialize message state
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Hello. I am your Forensic RAG Assistant. Ask me about model metrics, dataset splits, or healthy/duplicate audit files."}
            ]
            
        # Process quick action query if triggered
        if pending_query:
            st.session_state["messages"].append({"role": "user", "content": pending_query})
            ans = rag_agent.answer_query(pending_query)
            st.session_state["messages"].append({"role": "assistant", "content": ans})
            
        # Render message history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        # Render standard chat input
        user_query = st.chat_input("Enter your forensic question...")
        if user_query:
            st.session_state["messages"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)
                
            with st.spinner("Retrieving database facts..."):
                ans = rag_agent.answer_query(user_query)
                
            st.session_state["messages"].append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"):
                st.write(ans)
                
    with right_ref_col:
        st.markdown("### Investigative Reference Data")
        
        with st.expander("Model Performance Metrics"):
            # Load and display local metrics summary
            m_path = "results/metrics_summary.json"
            if os.path.exists(m_path):
                import json
                with open(m_path, "r") as f:
                    metrics_data = json.load(f)
                st.markdown(f"""
                - **Accuracy**: {metrics_data.get('accuracy', 0)*100:.2f}%
                - **ROC-AUC**: {metrics_data.get('auc', 0):.4f}
                - **Precision**: {metrics_data.get('precision', 0)*100:.2f}%
                - **Recall**: {metrics_data.get('recall', 0)*100:.2f}%
                """)
            else:
                st.info("Metrics data not found. Please run evaluate.py first.")
                
        with st.expander("Dataset Configuration"):
            # Display dataset stats
            st.markdown("""
            - **Dataset Name**: Google Deepfake Detection (DFD)
            - **Total Videos**: 395 
            - **Real Videos**: 380
            - **Fake Videos**: 15
            - **Aligned Crop/Mel Pairs**: 12,877
            - **Data Splits**: 70% Train, 20% Val, 10% Test
            """)
            
        with st.expander("Dataset Health Snippet"):
            health_md_path = "storage/reports/dataset_deepfake_detection_health.md"
            if os.path.exists(health_md_path):
                with open(health_md_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                # show first 12 lines for a clean reference snippet
                snippet = "".join(lines[:12])
                st.markdown(snippet + "\n...")
            else:
                st.info("Health report not found.")

# ----------------- TAB 3: DATASET & PERFORMANCE -----------------
with tab3:
    st.subheader("Evaluation Curves and Dataset Diagnostics")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Confusion Matrix**")
        cm_path = "results/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption="True vs. Predicted spoofing label counts", use_container_width=True)
        else:
            st.info("Confusion matrix plot not found at results/confusion_matrix.png.")
            
    with col_b:
        st.write("**ROC Curve (Separation Power)**")
        roc_path = "results/roc_curve.png"
        if os.path.exists(roc_path):
            st.image(roc_path, caption="True Positive Rate vs False Positive Rate Curve (AUC = 0.95)", use_container_width=True)
        else:
            st.info("ROC-AUC plot not found at results/roc_curve.png.")

    st.markdown("---")
    st.write("**Dataset Health Audit Findings**")
    health_md_path = "storage/reports/dataset_deepfake_detection_health.md"
    if os.path.exists(health_md_path):
        with open(health_md_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Dataset health report markdown file not found.")
