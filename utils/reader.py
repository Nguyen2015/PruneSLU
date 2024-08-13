import json
import os
import random
import sys
from typing import List

import librosa
import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.binary import DatasetReader


class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 processor,
                 mono=True,
                 language=None,
                 timestamps=False,
                 sample_rate=16000,
                 min_duration=0.5,
                 max_duration=30,
                 augment_config_path=None):
        """
        Args:
             data_list_path: The path to the data list file, or the header file path of the binary list
             processor: Whisper's preprocessing tool, obtained by WhisperProcessor.from_pretrained
             mono: Whether to convert the audio to a single channel, this must be True
             language: the language of the fine-tuned data
             timestamps: whether to use timestamps when fine-tuning
             sample_rate: audio sampling rate, default is 16000
             min_duration: Audio less than this time period will be truncated, unit second, cannot be less than 0.5, default is 0.5s
             max_duration: Audio greater than this time period will be truncated, in seconds, cannot be greater than 30, the default is 30s
             augment_config_path: Data augmentation configuration parameter file path
        """
        super(CustomDataset, self).__init__()
        assert min_duration >= 0.5, f"min_duration不能小于0.5，当前为：{min_duration}"
        assert max_duration <= 30, f"max_duration不能大于30，当前为：{max_duration}"
        self.data_list_path = data_list_path
        self.processor = processor
        self.data_list_path = data_list_path
        self.sample_rate = sample_rate
        self.mono = mono
        self.language = language
        self.timestamps = timestamps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.vocab = self.processor.tokenizer.get_vocab()
        self.startoftranscript = self.vocab['<|startoftranscript|>']
        self.endoftext = self.vocab['<|endoftext|>']
        if '<|nospeech|>' in self.vocab.keys():
            self.nospeech = self.vocab['<|nospeech|>']
            self.timestamp_begin = None
        else:
            # 兼容旧模型
            self.nospeech = self.vocab['<|nocaptions|>']
            self.timestamp_begin = self.vocab['<|notimestamps|>'] + 1
        self.data_list: List[dict] = []
        # 加载数据列表
        self._load_data_list()
        # 数据增强配置参数
        self.augment_configs = None
        self.noises_path = None
        self.speed_rates = None
        if augment_config_path:
            with open(augment_config_path, 'r', encoding='utf-8') as f:
                self.augment_configs = json.load(f)

    # 加载数据列表
    def _load_data_list(self):
        if self.data_list_path.endswith(".header"):
            # 获取二进制的数据列表
            self.dataset_reader = DatasetReader(data_header_path=self.data_list_path,
                                                min_duration=self.min_duration,
                                                max_duration=self.max_duration)
            self.data_list = self.dataset_reader.get_keys()
        else:
            # 获取数据列表
            with open(self.data_list_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            self.data_list = []
            for line in tqdm(lines, desc='读取数据列表'):
                if isinstance(line, str):
                    line = json.loads(line)
                if not isinstance(line, dict): 
                    print(line)
                    continue
                # 跳过超出长度限制的音频
                if line["duration"] < self.min_duration:
                    continue
                if self.max_duration != -1 and line["duration"] > self.max_duration:
                    continue
                self.data_list.append(dict(line))
                

    # 从数据列表里面获取音频数据、采样率和文本
    def _get_list_data(self, idx):
        if self.data_list_path.endswith(".header"):
            data_list = self.dataset_reader.get_data(self.data_list[idx])
        else:
            data_list = self.data_list[idx]
        # 分割音频路径和标签
        audio_file = data_list["audio"]['path']
        transcript = data_list["sentences"] if self.timestamps else data_list["sentence"]
        language = data_list["language"] if 'language' in data_list.keys() else None
        if 'start_time' not in data_list["audio"].keys():
            sample, sample_rate = soundfile.read(audio_file, dtype='float32')
        else:
            start_time, end_time = data_list["audio"]["start_time"], data_list["audio"]["end_time"]
            # 分割读取音频
            sample, sample_rate = self.slice_from_file(audio_file, start=start_time, end=end_time)
        sample = sample.T
        # 转成单通道
        if self.mono:
            sample = librosa.to_mono(sample)
        # 数据增强
        if self.augment_configs:
            sample, sample_rate = self.augment(sample, sample_rate)
        # 重采样
        if self.sample_rate != sample_rate:
            sample = self.resample(sample, orig_sr=sample_rate, target_sr=self.sample_rate)
        return sample, sample_rate, transcript, language

    def _load_timestamps_transcript(self, transcript: List[dict]):
        assert isinstance(transcript, list), f"transcript应该为list，当前为：{type(transcript)}"
        data = dict()
        labels = self.processor.tokenizer.prefix_tokens[:3]
        for t in transcript:
            # 将目标文本编码为标签ID
            start = t['start'] if round(t['start'] * 100) % 2 == 0 else t['start'] + 0.01
            if self.timestamp_begin is None:
                start = self.vocab[f'<|{start:.2f}|>']
            else:
                start = self.timestamp_begin + round(start * 100) // 2
            end = t['end'] if round(t['end'] * 100) % 2 == 0 else t['end'] - 0.01
            if self.timestamp_begin is None:
                end = self.vocab[f'<|{end:.2f}|>']
            else:
                end = self.timestamp_begin + round(end * 100) // 2
            label = self.processor(text=t['text']).input_ids[4:-1]
            labels.extend([start])
            labels.extend(label)
            labels.extend([end])
        data['labels'] = labels + [self.endoftext]
        return data

    def __getitem__(self, idx):
        # try:
            # Get audio data, sample rate, and text from the data list
            sample, sample_rate, transcript, language = self._get_list_data(idx=idx)
            # You can set the language for individual data
            self.processor.tokenizer.set_prefix_tokens(language=language if language is not None else self.language)
            if len(transcript) > 0:
                # Load text with timestamps
                if self.timestamps:
                    data = self._load_timestamps_transcript(transcript=transcript)
                    # Calculate log-Mel input features from the input audio array
                    data["input_features"] = self.processor(audio=sample, sampling_rate=self.sample_rate).input_features
                else:
                    # Get log-Mel features and label IDs
                    data = self.processor(audio=sample, sampling_rate=self.sample_rate, text=transcript)
            else:
                # If there is no text, use the <|nospeech|> tag
                data = self.processor(audio=sample, sampling_rate=self.sample_rate)
                data['labels'] = [self.startoftranscript, self.nospeech, self.endoftext]
            return data
        # except Exception as e:
        #     print(f'Error reading data, sequence number: {idx}, error message: {e}', file=sys.stderr)
        #     return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.data_list)
    
    # Slice and read audio
    @staticmethod
    def slice_from_file(file, start, end):
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = round(float(len(sndfile)) / sample_rate, 3)
        start = round(start, 3)
        end = round(end, 3)
        # Start counting from the end
        if start < 0.0: start += duration
        if end < 0.0: end += duration
        # Ensure data does not exceed bounds
        if start < 0.0: start = 0.0
        if end > duration: end = duration
        if end < 0.0:
            raise ValueError("Slice end position (%f s) out of bounds" % end)
        if start > end:
            raise ValueError("Slice start position (%f s) later than slice end position (%f s)" % (start, end))
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        sample = sndfile.read(frames=end_frame - start_frame, dtype='float32')
        return sample, sample_rate

    # Data augmentation
    def augment(self, sample, sample_rate):
        for config in self.augment_configs:
            if config['type'] == 'speed' and random.random() < config['prob']:
                if self.speed_rates is None:
                    min_speed_rate, max_speed_rate, num_rates = config['params']['min_speed_rate'], \
                        config['params']['max_speed_rate'], config['params']['num_rates']
                    self.speed_rates = np.linspace(min_speed_rate, max_speed_rate, num_rates, endpoint=True)
                rate = random.choice(self.speed_rates)
                sample = self.change_speed(sample, speed_rate=rate)
            if config['type'] == 'shift' and random.random() < config['prob']:
                min_shift_ms, max_shift_ms = config['params']['min_shift_ms'], config['params']['max_shift_ms']
                shift_ms = random.randint(min_shift_ms, max_shift_ms)
                sample = self.shift(sample, sample_rate, shift_ms=shift_ms)
            if config['type'] == 'volume' and random.random() < config['prob']:
                min_gain_dBFS, max_gain_dBFS = config['params']['min_gain_dBFS'], config['params']['max_gain_dBFS']
                gain = random.randint(min_gain_dBFS, max_gain_dBFS)
                sample = self.volume(sample, gain=gain)
            if config['type'] == 'resample' and random.random() < config['prob']:
                new_sample_rates = config['params']['new_sample_rates']
                new_sample_rate = np.random.choice(new_sample_rates)
                sample = self.resample(sample, orig_sr=sample_rate, target_sr=new_sample_rate)
                sample_rate = new_sample_rate
            if config['type'] == 'noise' and random.random() < config['prob']:
                min_snr_dB, max_snr_dB = config['params']['min_snr_dB'], config['params']['max_snr_dB']
                if self.noises_path is None:
                    self.noises_path = []
                    noise_dir = config['params']['noise_dir']
                    if os.path.exists(noise_dir):
                        for file in os.listdir(noise_dir):
                            self.noises_path.append(os.path.join(noise_dir, file))
                noise_path = random.choice(self.noises_path)
                snr_dB = random.randint(min_snr_dB, max_snr_dB)
                sample = self.add_noise(sample, sample_rate, noise_path=noise_path, snr_dB=snr_dB)
        return sample, sample_rate

    # 改变语速
    @staticmethod
    def change_speed(sample, speed_rate):
        if speed_rate == 1.0:
            return sample
        if speed_rate <= 0:
            raise ValueError("速度速率应大于零")
        old_length = sample.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        sample = np.interp(new_indices, old_indices, sample).astype(np.float32)
        return sample

    # 音频偏移
    @staticmethod
    def shift(sample, sample_rate, shift_ms):
        duration = sample.shape[0] / sample_rate
        if abs(shift_ms) / 1000.0 > duration:
            raise ValueError("shift_ms的绝对值应该小于音频持续时间")
        shift_samples = int(shift_ms * sample_rate / 1000)
        if shift_samples > 0:
            sample[:-shift_samples] = sample[shift_samples:]
            sample[-shift_samples:] = 0
        elif shift_samples < 0:
            sample[-shift_samples:] = sample[:shift_samples]
            sample[:-shift_samples] = 0
        return sample

    # 改变音量
    @staticmethod
    def volume(sample, gain):
        sample *= 10.**(gain / 20.)
        return sample

    # 声音重采样
    @staticmethod
    def resample(sample, orig_sr, target_sr):
        sample = librosa.resample(sample, orig_sr=orig_sr, target_sr=target_sr)
        return sample

    # 添加噪声
    def add_noise(self, sample, sample_rate, noise_path, snr_dB, max_gain_db=300.0):
        noise_sample, sr = librosa.load(noise_path, sr=sample_rate)
        # 标准化音频音量，保证噪声不会太大
        target_db = -20
        gain = min(max_gain_db, target_db - self.rms_db(sample))
        sample *= 10. ** (gain / 20.)
        # 指定噪声音量
        sample_rms_db, noise_rms_db = self.rms_db(sample), self.rms_db(noise_sample)
        noise_gain_db = min(sample_rms_db - noise_rms_db - snr_dB, max_gain_db)
        noise_sample *= 10. ** (noise_gain_db / 20.)
        # 固定噪声长度
        if noise_sample.shape[0] < sample.shape[0]:
            diff_duration = sample.shape[0] - noise_sample.shape[0]
            noise_sample = np.pad(noise_sample, (0, diff_duration), 'wrap')
        elif noise_sample.shape[0] > sample.shape[0]:
            start_frame = random.randint(0, noise_sample.shape[0] - sample.shape[0])
            noise_sample = noise_sample[start_frame:sample.shape[0] + start_frame]
        sample += noise_sample
        return sample

    @staticmethod
    def rms_db(sample):
        mean_square = np.mean(sample ** 2)
        return 10 * np.log10(mean_square)
        