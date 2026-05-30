# whisper-transcribe-api

API FastAPI para transcrição de vídeos e áudios de reuniões, usando **Faster-Whisper** com aceleração GPU e pipeline de pré-processamento via **FFmpeg**.

## Stack

- **Faster-Whisper** — reimplementação do Whisper com CTranslate2 (4× mais rápido, menos VRAM)
- **FastAPI** — API REST para upload e transcrição
- **FFmpeg** — extração de áudio + speed-up 2.5× antes do Whisper
- **uv** — gerenciamento de dependências

## Requisitos

- Python 3.12+
- FFmpeg instalado no sistema
- GPU NVIDIA (recomendado) — roda em CPU também, mas mais lento

```bash
# Ubuntu/Debian
sudo apt install ffmpeg -y
```

## Instalação

```bash
# Clone o repositório
git clone https://github.com/WagnerVic/whisper-transcribe-api
cd whisper-transcribe-api

# Criar ambiente e instalar dependências
uv sync

# Ativar o ambiente
source .venv/bin/activate
```

## Uso

### Iniciar a API

```bash
# Via entry point
python main.py

# Ou com uvicorn diretamente
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

A API estará disponível em `http://localhost:8000`.
Documentação interativa: `http://localhost:8000/docs`

### Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET`  | `/health` | Status da API e dispositivo em uso |
| `POST` | `/transcribe` | Upload de vídeo/áudio → transcrição |

### Transcrever um vídeo

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@reuniao.mp4" \
  -F "language=pt"
```

**Parâmetros do `/transcribe`:**

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `file` | file | obrigatório | Vídeo ou áudio |
| `language` | string | auto | Código do idioma (`pt`, `en`, `es`...) |
| `speed_up` | float | `2.5` | Fator de aceleração FFmpeg antes da transcrição |

### Formatos suportados

**Vídeo:** `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`, `.wmv`
**Áudio:** `.mp3`, `.wav`, `.flac`, `.ogg`, `.aac`, `.m4a`, `.opus`

### Saída

Os arquivos são salvos automaticamente na pasta `output/`:

| Formato | Conteúdo |
|---------|----------|
| `.txt` | Texto corrido, sem timestamps |
| `.srt` | Legendas com timestamps (padrão para VLC, YouTube) |
| `.json` | Estrutura completa com segmentos, timestamps por palavra e metadados |

A resposta JSON da API também retorna tudo em tempo real.

## Modelos Disponíveis

| Modelo | VRAM | Velocidade | Precisão |
|--------|------|-----------|----------|
| `tiny` | ~1 GB | ⚡⚡⚡⚡⚡ | ★★☆☆☆ |
| `base` | ~1 GB | ⚡⚡⚡⚡ | ★★★☆☆ |
| `small` | ~2 GB | ⚡⚡⚡ | ★★★☆☆ |
| `medium` | ~5 GB | ⚡⚡ | ★★★★☆ |
| `large-v3` | ~5-6 GB | ⚡ | ★★★★★ |
| `turbo` | ~6 GB | ⚡⚡⚡ | ★★★★☆ |

**Default:** `large-v3` com quantização `int8_float16` (máxima precisão para GPUs com 6+ GB VRAM).

## Pipeline FFmpeg

O áudio passa por um pipeline antes da transcrição para reduzir o tempo de processamento em ~60%:

1. **Extração de áudio** (se for vídeo) → WAV mono 16 kHz
2. **Speed-up 2.5×** via filtro `atempo` — mantém qualidade adequada para transcrição
3. **Bitrate 64 kbps** — reduz tamanho sem impactar precisão

Os timestamps na resposta são automaticamente ajustados para refletir o tempo real do arquivo original.

## Estrutura do Projeto

```
whisper-transcribe-api/
├── app/
│   ├── __init__.py
│   ├── audio_processor.py   # Pipeline FFmpeg
│   ├── transcriber.py       # Motor Faster-Whisper
│   ├── schemas.py           # Pydantic models
│   └── main.py              # FastAPI + rotas
├── output/                  # Transcrições geradas
├── uploads/                 # Uploads temporários
├── main.py                  # Entry point
├── pyproject.toml
└── README.md
```
