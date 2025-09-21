There have been reports of attempts to improve accuracy by modifying the loss function of SepFormer to resemble that of SepReformer.

# A report on an attempt to improve accuracy by modifying the loss function of SepFormer to resemble that of SepReformer.

## Motivation for attempting to improve SepFormer

I learned that machine learning is also producing results in the field of audio source separation, so I started studying it. I created the Libri2Mix data and first tried running Conv-TasNet. After learning, I played back the results and confirmed that it was trying to create separate wav files for each of the two speakers. So I moved on to a sound source separation program using a Transformer. I got SepFormer and SepReformer to work. The training results showed that SepFormer's Si-SNR loss was -13.59, and SepReformer's Si-SNRi was 13.68. When I played back the separated files, SepReformer seemed to have slightly better sound quality than SepFormer. So I decided to try modifying SepFormer.

## Improvement points

The modifications were limited to the loss function. In SepFormer, the Si-SNR value multiplied by -1 was used as the loss function. In SepReformer, the average of PIT_SI_SNR_mag and PIT_SI_SNR_time of the four stages was used. We decided to use the PIT_SI_SNR_mag class and PIT_SI_SNR_time class. Therefore, I changed masknet_numlayers in SepFormer's masknet, changing self.dual_mdl from 2 layers to 4 layers, and post-processing was performed on each layer, resulting in a 4-layer masknet output. The 4-layer est_source_layers was calculated from these outputs. The PIT_SI_SNR_time of the 4-layer est_source_layers was calculated and averaged. I also calculated PIT_SI_SNR_mag from est_source_layers[-1]. The sum of these two was used as the loss.

## Results

<table>
<caption> Si-SNR and SDR
<thread>
<th>model<th>Si-SNRi<th>SDR
<tbody>
<tr><td style="text-align:left;"> Original SepFormer<td>-13.59<td>
<tr><td style="text-align:left;"> SepReformer<td>13.68<td>14.16
<tr><td style="text-align:left;"> Modified SepFormer<td>14.74<td>15.58
</table>

Libri2Mix 8k min sep-clean train-100 data with batch_size = 1 and epoch = 10.
