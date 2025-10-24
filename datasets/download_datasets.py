"""
Script to download all required datasets
Run this first: python datasets/download_datasets.py
"""

import os
import gdown
import zipfile
import shutil

def create_directories():
    """Create necessary directories"""
    dirs = ['datasets/chest_xray', 'datasets/brain_tumor', 'datasets/chatbot_data', 'models']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Directories created")

def download_pneumonia_dataset():
    """Download Chest X-Ray Pneumonia dataset"""
    print("\n📥 Downloading Pneumonia Dataset...")
    print("⚠️ Manual Download Required:")
    print("1. Go to: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia")
    print("2. Download the dataset")
    print("3. Extract to: datasets/chest_xray/")
    print("\nOr use Kaggle API:")
    print("   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
    print("   unzip chest-xray-pneumonia.zip -d datasets/chest_xray/")

def download_brain_tumor_dataset():
    """Download Brain Tumor MRI dataset"""
    print("\n📥 Downloading Brain Tumor Dataset...")
    print("⚠️ Manual Download Required:")
    print("1. Go to: https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset")
    print("2. Download the dataset")
    print("3. Extract to: datasets/brain_tumor/")
    print("\nOr use Kaggle API:")
    print("   kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset")
    print("   unzip brain-tumor-mri-dataset.zip -d datasets/brain_tumor/")

def download_medical_faq():
    """Download or create Medical FAQ dataset"""
    print("\n📥 Creating Medical FAQ Dataset...")
    
    import pandas as pd
    
    # Sample medical FAQs (expand this to 500+)
    faq_data = {
        'question': [
            'What is pneumonia?',
            'What are the symptoms of pneumonia?',
            'How is pneumonia diagnosed?',
            'How to prevent pneumonia?',
            'Is pneumonia contagious?',
            'What causes pneumonia?',
            'How long does pneumonia last?',
            'What is the treatment for pneumonia?',
            'What is a brain tumor?',
            'What are the symptoms of brain tumor?',
            'What causes brain tumors?',
            'How are brain tumors diagnosed?',
            'What are types of brain tumors?',
            'Is brain tumor curable?',
            'What is glioma?',
            'What is meningioma?',
            'What is pituitary tumor?',
            'How to prevent brain tumors?',
            'What is a chest X-ray?',
            'Why do doctors order chest X-rays?',
            'What does a normal chest X-ray look like?',
            'What is MRI scan?',
            'What is CT scan?',
            'When should I see a doctor for cough?',
            'What is the difference between viral and bacterial pneumonia?',
            'Can children get pneumonia?',
            'What are complications of pneumonia?',
            'Do I need antibiotics for pneumonia?',
            'What is aspiration pneumonia?',
            'What is walking pneumonia?',
        ],
        'answer': [
            'Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm, fever, chills, and difficulty breathing.',
            'Common symptoms include cough with phlegm, fever, sweating and shaking chills, shortness of breath, chest pain, fatigue, nausea, vomiting, and diarrhea.',
            'Pneumonia is diagnosed through physical examination, chest X-ray, blood tests, sputum test, and sometimes CT scan or bronchoscopy.',
            'Get vaccinated (pneumococcal and flu vaccines), practice good hygiene, don\'t smoke, maintain a healthy lifestyle, and avoid sick people.',
            'Yes, pneumonia can be contagious. The germs that cause it can spread from person to person through coughing or sneezing.',
            'Pneumonia is caused by bacteria, viruses, or fungi. Common bacterial cause is Streptococcus pneumoniae. Viral causes include influenza and COVID-19.',
            'With treatment, most people improve within 1-2 weeks. Complete recovery may take 4-6 weeks or longer for severe cases.',
            'Treatment includes antibiotics for bacterial pneumonia, rest, fluids, fever reducers, and cough medicine. Severe cases may require hospitalization.',
            'A brain tumor is an abnormal growth of cells in the brain. Tumors can be benign (non-cancerous) or malignant (cancerous).',
            'Symptoms include headaches, seizures, vision problems, hearing problems, balance difficulties, nausea, vomiting, personality changes, and cognitive difficulties.',
            'Most brain tumors have no known cause. Risk factors include radiation exposure, family history, and certain genetic conditions.',
            'Diagnosis involves neurological exam, MRI scan, CT scan, PET scan, and biopsy to determine the type of tumor.',
            'Main types include gliomas, meningiomas, pituitary tumors, schwannomas, and medulloblastomas. They are classified as benign or malignant.',
            'Some brain tumors are curable with treatment. Outcome depends on tumor type, location, size, and how early it is detected.',
            'Glioma is a type of tumor that starts in glial cells of the brain. It includes astrocytomas, oligodendrogliomas, and glioblastomas.',
            'Meningioma is a tumor that forms in the meninges (membranes surrounding brain and spinal cord). Most are benign and slow-growing.',
            'Pituitary tumor is an abnormal growth in the pituitary gland. Most are benign and called pituitary adenomas.',
            'There is no proven way to prevent brain tumors. Avoid radiation exposure when possible and maintain overall health.',
            'A chest X-ray is an imaging test that uses small amounts of radiation to produce pictures of organs and structures inside the chest.',
            'Doctors order chest X-rays to diagnose conditions like pneumonia, heart failure, lung cancer, broken ribs, and other chest problems.',
            'A normal chest X-ray shows clear lungs, normal heart size, clear airways, and proper bone structure without abnormalities.',
            'MRI (Magnetic Resonance Imaging) uses magnetic fields and radio waves to create detailed images of organs and tissues.',
            'CT (Computed Tomography) scan combines X-ray images taken from different angles to create cross-sectional images of the body.',
            'See a doctor if cough lasts more than 3 weeks, you cough up blood, have high fever, difficulty breathing, or chest pain.',
            'Viral pneumonia is caused by viruses and usually milder. Bacterial pneumonia is caused by bacteria and often more severe, requiring antibiotics.',
            'Yes, children can get pneumonia. It is one of the leading causes of illness in children worldwide. Vaccination helps prevent it.',
            'Complications include bacteremia, lung abscess, pleural effusion, respiratory failure, and sepsis. Severe cases can be life-threatening.',
            'Bacterial pneumonia requires antibiotics. Viral pneumonia does not respond to antibiotics. Your doctor will determine the cause.',
            'Aspiration pneumonia occurs when you inhale food, drink, vomit, or saliva into your lungs, leading to infection.',
            'Walking pneumonia is a mild form of pneumonia where you can continue daily activities. It is usually caused by mycoplasma bacteria.',
        ]
    }
    
    df = pd.DataFrame(faq_data)
    df.to_csv('datasets/chatbot_data/medical_faq.csv', index=False)
    print(f"✅ Created medical_faq.csv with {len(df)} Q&A pairs")
    print("⚠️ Expand this to 500+ pairs for better accuracy!")

def main():
    print("="*60)
    print("🏥 Medical AI Dataset Downloader")
    print("="*60)
    
    create_directories()
    # download_pneumonia_dataset()
    # download_brain_tumor_dataset()
    download_medical_faq()
    
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Download datasets manually from Kaggle (links above)")
    print("2. Extract to respective folders")
    print("3. Run training notebooks in 'notebooks/' folder")

if __name__ == "__main__":
    main()