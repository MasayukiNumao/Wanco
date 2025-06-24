import asyncio
import websockets
import subprocess

import requests
import json
import sounddevice as sd
import numpy as np

def vvox_tts(text):
    # エンジン起動時に表示されているIP、portを指定
    host = "localhost"
    port = 50021
    
    # 音声化する文言と話者を指定(3で標準ずんだもんになる)
    params = (
        ('text', text),
        ('speaker', 3),
    )
    
    # 音声合成用のクエリ作成
    query = requests.post(
        f'http://{host}:{port}/audio_query',
        params=params
    )
    
    # 音声合成を実施
    synthesis = requests.post(
        f'http://{host}:{port}/synthesis',
        headers = {"Content-Type": "application/json"},
        params = params,
        data = json.dumps(query.json())
    )
    
    # 音声データを取得
    voice = synthesis.content

    # 音声データをNumPy配列に変換（16ビットPCMに対応）
    voice_data = np.frombuffer(voice, dtype=np.int16)
    
    # サンプリングレートが24000以外だとずんだもんが高音になったり低音になったりする
    sample_rate = 24000

    # 再生処理 (sounddeviceを使用)
    sd.play(voice_data, samplerate=sample_rate)
    sd.wait()  # 再生が終わるまで待機

async def handle_connection(websocket, path):
    async for message in websocket:
        print(f"Received message: {message}")
        # Call Aquestalk TTS
        vvox_tts(message)
        # Send completion signal
        await websocket.send("TTS complete")

async def main():
    async with websockets.serve(handle_connection, "localhost", 8766):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
