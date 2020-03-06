# We retain the copyright notice by NVIDIA from the original code. However, we
# we reserve our rights on the modifications based on the original code.
#
# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import glob
import os
import numpy as np
import torch
from denoiser import Denoiser
from scipy.io.wavfile import write


MAX_WAV_VALUE = 32768

def main(squeezewave_path, sigma, output_dir, sampling_rate, is_fp16, denoiser_strength):
    squeezewave = torch.load(squeezewave_path)['model']
    squeezewave = squeezewave.remove_weightnorm(squeezewave)
    squeezewave.cuda().eval()
    if is_fp16:
        from apex import amp
        squeezewave, _ = amp.initialize(squeezewave, [], opt_level="O3")

    if denoiser_strength > 0:
        denoiser = Denoiser(squeezewave).cuda()

    for i, file_path in enumerate(glob.glob('*.npy')):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        mel = torch.from_numpy(np.load(file_path))
        mel = torch.unsqueeze(mel, 0).cuda()
        mel = mel.half() if is_fp16 else mel
        with torch.no_grad():
            audio = squeezewave.infer(mel, sigma=sigma)
            if denoiser_strength > 0:
                audio = denoiser(audio, denoiser_strength)
            print(audio.min(), audio.max())
            audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze().cpu().numpy()
        audio_path = os.path.join(output_dir, 'SqueezeWave_{}.wav'.format(file_name))
        write(audio_path, sampling_rate, audio.astype('int16'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--squeezewave_path',
                        help='Path to squeezewave decoder checkpoint with model')
    parser.add_argument('-o', "--output_dir", default='.')
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    main(args.squeezewave_path, args.sigma, args.output_dir,
         args.sampling_rate, args.is_fp16, args.denoiser_strength)
