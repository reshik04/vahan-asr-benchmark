# ASR Benchmarking for Indian Conversational Speech
> Vahan AI Intern Assessment — Reshik Rabacca

## 🎯 Problem
Vahan's hiring platform serves blue-collar workers who interact via phone calls in Hindi/Hinglish, in noisy environments. The core challenge: **can ASR systems correctly extract Bangalore locality names from conversational speech?**

## 🔬 Models Benchmarked
| Model | Type | Locality Accuracy | Avg WER | Avg Latency |
|-------|------|-------------------|---------|-------------|
| Deepgram Nova-2 | API (baseline) | 10% | 0.900 | 2.63s |
| Whisper Medium | Local OSS | 15% | 0.995 | 6.42s |
| Wav2Vec2-XLSR-Hindi | Local OSS | 0% | 1.051 | 0.41s |

## 💡 Key Finding
**No ASR system reliably extracts Bangalore locality names out of the box.**
The real solution isn't which ASR model you pick — it's building a **fuzzy matching post-processor** that maps ASR output to a known locality list:
"हिटलर लेट" → fuzzy match → "HSR Layout" ✓
"बंगला"     → fuzzy match → "Koramangala" ✓
"मनाली"     → fuzzy match → "Bommanahalli" ✓

## 📊 Dataset
- 20 self-recorded audio samples of Bangalore locality names
- Natural Hindi/Hinglish conversational sentences
- 4 conditions: quiet room, background noise, rushed speech, whispered
- Recorded on phone mic to simulate real candidate calls

## 📁 Structure
```
vahan_asr_benchmark/
├── recordings/          # 20 wav files across 4 conditions
├── results/             # raw transcripts + full evaluated results
├── report/              # 4 charts + written report
├── pipeline.py          # runs all 3 ASR models
├── metrics.py           # WER + locality accuracy calculation
├── charts.py            # generates all visualizations
└── ground_truth.csv     # reference transcriptions
```

## 🚀 Setup & Run
```bash
# 1. Clone
git clone https://github.com/reshik04/vahan-asr-benchmark.git
cd vahan-asr-benchmark

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Deepgram API key
cp .env.example .env
# edit .env and add your key

# 5. Run pipeline
python pipeline.py

# 6. Calculate metrics
python metrics.py

# 7. Generate charts
python charts.py
```

## 📈 Results

### WER by Model
![WER by Model](report/chart1_wer_by_model.png)

### Locality Accuracy
![Locality Accuracy](report/chart2_locality_accuracy.png)

### Performance by Condition
![By Condition](report/chart3_wer_by_condition.png)

### Latency Comparison
![Latency](report/chart4_latency.png)

## 🔍 Failure Analysis Highlights
| Ground Truth | Deepgram | Whisper |
|-------------|----------|---------|
| HSR Layout | हिटलर लेट (Hitler Late!) | hedge sir layout |
| Koramangala | बंगला | मंगला |
| Bommanahalli | मनाली | उमना हली |
| Banashankari | माना शंकर | मानाशंकरी |

## ✅ Recommendation
**Deepgram Nova-2 + fuzzy locality post-processor** for production.
- Best WER and locality accuracy among tested models
- API-based = no infra overhead
- Whisper viable at scale (>1M calls/month) despite higher latency
- Wav2Vec2 not production-ready in current form

## 📹 Walkthrough Video
[Watch the 6-minute project walkthrough here](https://drive.google.com/drive/folders/1-lSz1eFCO4lYvxxrIkWM0kMZ3pRGeevk) 
