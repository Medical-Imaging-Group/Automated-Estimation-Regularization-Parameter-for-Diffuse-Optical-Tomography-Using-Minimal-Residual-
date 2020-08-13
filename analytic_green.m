function [time,mono_time] = analytic_green(srx,rxs,maxfreq,nfreq,c)
%Input
%      srx     -  location of sources
%      rxs     -  location of receivers 
%      maxfreq -  maximum frequency 
%      nfreq   -  number of frequencies to model
%      c       -  velocity of the medium 

nsrx=size(srx,1); % number of sources
nrxs=size(rxs,1); % number of receivers
nf = 2^nextpow2(nfreq); % round number of frequencies to the nearest power of two to make fft/ifft run faster

% Calculate df and frequency range
df = maxfreq/nf;
freq = (1:nf).*df;
freq=reshape(freq,nf,1);
om = 2*pi*freq; % angular frequency = 2*pi*frequency
kp = om./c; % wavenumber = angular frequency / velocity

% Calculate dt and time range
dt = 1/(2*maxfreq); % time step
time = dt.*(0:2*nf-1);
time = reshape(time,2*nf,1);

offs = zeros(nsrx,nrxs);
for ss = 1:nsrx
    offs(ss,1:nrxs) = sqrt((srx(ss,1)-rxs(1:nrxs,1)).^2+(srx(ss,2)-rxs(1:nrxs,2)).^2);
end
nf = length(freq); % number of frequencies
% Allocate space for results
u_mon_p_f = zeros(nf,nrxs,nsrx);
% Calculate Green's function
for ss = 1:nsrx
    gf = -1i./4.*besselh(0,2,repmat(kp,[1 nrxs]).*repmat(offs(ss,:),[nf 1]));
    u_mon_p_f(:,:,ss) = 1i.*(repmat(om,[1 nrxs])).*gf;
    %u_mon_p_f(:,:,ss) = gf;
end

gf_mon_p_f1nf = zeros(nf,nrxs,nsrx);
gf_mon_p_f1nf(:,1:nrxs,1:nsrx) = u_mon_p_f(1:nf,1:nrxs,1:nsrx);
gf_mon_p_ff = fftshift([zeros(1,size(gf_mon_p_f1nf,2),size(gf_mon_p_f1nf,3));...
     gf_mon_p_f1nf(1:nf-1,:,:)./sqrt(2);real(gf_mon_p_f1nf(nf,:,:));flipdim(conj(gf_mon_p_f1nf(1:nf-1,:,:)),1)./sqrt(2)],1);
 % Applying inverse FFT to obtain response in time domain
mono_time = real(sqrt(2*nf).*ifft(fftshift(gf_mon_p_ff,1),[],1));
