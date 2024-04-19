%%  WOLA implementation of fully connected DANSE_1 and rS-DANSE_1 for 
%%  multi-channel signal enhancement (single desired source)

%%  Version: 1.4

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright (c) 2010, Alexander Bertrand
% All rights reserved.
% Redistribution and use in source and binary forms, with or
% without modification, are permitted provided that the
% following conditions are met:
% 1) Redistributions of source code must retain the above copyright
% notice and this list of conditions.
% 2) Redistributions in binary form must reproduce the above copyright
% notice and this list of conditions in the documentation and/or other
% materials provided with the distribution.
% 3) The name of its contributors may not be used to endorse or promote
% products derived from this software without specific prior written
% permission.
%
% Contact: alexander.bertrand@esat.kuleuven.be (please report bugs)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  More info can be found in the paper

%  A. Bertrand, J. Callebaut and M. Moonen, "Adaptive distributed noise
%  reduction for speech enhancement in wireless acoustic sensor networks",
%  Proc. of the International Workshop on Acoustic Echo and Noise Control
%  (IWAENC), Tel Aviv, Israel, Aug. 2010.
%
%  and
%
%  Szurley J., Bertrand A., Moonen M., "Improved Tracking Performance for 
%  Distributed node-specific signal enhancement in wireless acoustic sensor
%  networks", 2013.

%  and on the website http://homes.esat.kuleuven.be/~abertran


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Remarks
% 1)    Optimal filtering is implemented with Rank-1 formula (assumes
% single speech source). Is theoretically equivalent to the DANSE_1 (or
% MWF) formula, but it is numerically favorable (see Cornelis, Moonen and
% Wouters, "Performance analysis of multi-channel Wiener filter based noise
% reduction in hearing aids under second order statistics estimation
% errors"). Note that convergence and optimality proofs of DANSE_1 remain
% applicable (for any value of mu).

% 2)    Nodes use a different filter to generate the broadcast signals
% (Wext) and to estimate the target speech (Wint). Wint is updated for each
% block of L samples, yielding faster local noise reduction. Wext is only
% updated during DANSE-updates. In that case, it will change Wext to the 
% current value of Wint. (see Szurley J., Bertrand A., Moonen M., "Improved
% Tracking Performance for Distributed node-specific signal enhancement in 
% wireless acoustic sensor networks", 2013.)

% 3)    The SDW-MWF parameter mu is hardcoded and should be >=0. mu=0
% corresponds to zero-distortion (equivalent to MVDR beamforming), mu=1
% corresponds to the original version of DANSE (i.e. MMSE estimation)

% 4)    All sensor signals in Y should be aligned such that the target 
% signal component in the different channels is no more than L/2 samples
% apart

% 5)    The first column of the Y{k}'s contains the reference microphone
% signals of the respective node

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% function [output,Ryy,Rnn,Wext,Wint,w]=WOLA_DANSE1(Y,fs,onoff,L,simultaneous,alpha,mu)

%% REQUIRED INPUTS:
%CELL ARRAY     Y            --> Y{k} is a NxM_k matrix containing the M_k microphone signals of node k in its columns (signals are N samples long)
%SCALAR         fs           --> Sample frequency of signals in Y
%VECTOR         onoff        --> Vector containing zeros and ones (should be same length as size(Y{k},1)), indicating with ones where target signal is active

%% OPTIONAL INPUTS (use '[]' to use default value):
%SCALAR         L            --> The FFTlength for WOLA (windowshift is L/2)
%SCALAR         simultaneous --> Set to 1 to let nodes update their parameters simultaneously (rS-DANSE). Set to 0 for sequential updates.
%SCALAR         alpha        --> The stepsize for relaxation (should be 0<=alpha<1) (if set to 1, there will be no relaxation)
%SCALAR         mu           --> SDW-MWF parameter, mu=0 corresponds to zero-distortion (=MVDR), mu=1 corresponds to unweighted MWF

%%   OUTPUTS: 
%CELL ARRAY     output       --> output{k} contains the output signal of node k
%CELL ARRAY     Ryy          --> Ryy{k} contains Signal+noise correlation matrices at node k, per frequency (in last iteration)
%CELL ARRAY     Rnn          --> Rnn{k} contains noise correlation matrices at node k, per frequency (in last iteration)
%CELL ARRAY     Wext         --> Wext{k} contains freq. domain (external) filters applied to the microphones of node k to generate the broadcast signal z_k
%CELL ARRAY     Wint         --> Wint{k} contains freq. domain (internal) filters applied to all signals available to node k, to generate the target signal estimate at node k
%CELL ARRAY     w            --> w{k} contains time domain (internal) filters applied to all signals available to node k, to generate the target signal estimate in the reference microphone of node k

%% Example input
% [output,Ryy,Rnn,Wext,Wint,w]=WOLA_DANSE1(Sensorsignals,16000,vad_signal,[],[],0.8,1)

%% NOTE: some parameters are hardcoded (see code for details):


function [output,Ryy,Rnn,Wext,Wint,w]=WOLA_DANSE1(Y,fs,onoff,varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  default values
L_default=512*fs/16000; %block length of 512 if sampling frequency is 16000
simultaneous_default=1; %use simultaneous updating by default
alpha_default=0.7; %note that some (different kind of) relaxation is already in place due to long-term memory of old data beyond the previous DANSE update (see 'lambda' and 'lambda_ext')
mu_default=1;

if nargin<3
    disp('NOT ENOUGH INPUT ARGUMENTS (at least 3 inputs are required)')
end
usevalues=[L_default simultaneous_default alpha_default mu_default];
T=length(varargin);

for j=1:T
    if ~isempty(varargin{j})
    usevalues(j)=varargin{j};
    end
end

L=usevalues(1);
simultaneous=usevalues(2);
alpha=usevalues(3);
mu=usevalues(4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   hardcoded parameters
nbsamples_per_sec=fs/(L/2); %number of DFT blocks per second

%Determines how fast older samples are forgotten:
lambda=exp(log(0.5)/(2*nbsamples_per_sec)); %forgettting factor for estimation of correlation matrices (should be 0<lambda<=1) (default: samples from 2 seconds in the past are weighted with 0.5)
lambda_ext=exp(log(0.5)/(0.2*nbsamples_per_sec)); %smoothing of external filter updates (set to zero for non-smooth updates, see code for more details)

%Determines update rate of DANSE
min_nb_samples=3*nbsamples_per_sec; %required number of new samples for both Ryy and Rnn before a new DANSE update can be performed (default: number of blocks in 3 seconds)

saveperiod=20; % every time this amount of seconds passed in the simulated signal, the results are saved
plotresults=1; %set to 1 if you want to plot results during simulation
shownode=1; %the node for which results are shown during simulation
fname='results'; %name for save file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   initialisation
nbnodes=length(Y); %number of nodes
lengthsignal=size(Y{1},1); %length of the audio signal
Ryysamples=0; %counts the number of samples used to estimate Ryy since last DANSE update
Rnnsamples=0; %counts the number of samples used to estimate Rnn since last DANSE update
for k=1:nbnodes
    nbmicsnode(k)=size(Y{k},2); %number of mics/sensors in node k
    dimnode(k)=nbmicsnode(k)+nbnodes-1; %dimension of the local estimation problem (for DANSE_1: =M_k+number of neighbors of node k)
    Wint{k}=zeros(dimnode(k),L/2+1); %filter internally applied to estimate signal
    Wint{k}(1,:)=1;
    Wext{k}=ones(nbmicsnode(k),L/2+1); %actual filter applied for broadcast signals (changes smoothly towards Wext_target{k}, based on exponential weighting with lambda_ext)
    Wext_target{k}=ones(nbmicsnode(k),L/2+1); %target filter applied for broadcast signals (at each DANSE-update of node k, Wext_target{k} is changed to Wint{k})
    for u=1:L/2+1
        Ryy{k}(:,:,u)=zeros(dimnode(k)); %signal covariance matrix
        Rnn{k}(:,:,u)=zeros(dimnode(k)); %noise covariance matrix (initialization)
    end
    estimation{k}=zeros(lengthsignal,1);
end

saveperiod=round(saveperiod*fs); %convert to number of samples passed
updatetoken=1; %the node that can perform the next update (only for sequential updating)
Han=hann(L)*ones(1,max(nbmicsnode)); %Hann-window (not equal to hanning)
startupdating=0; %flag to determine when the internal filters can start updating

teller=0; %counts percentage of signal that has been processed already
nbupdates=0; %counts number of DANSE updates that have been performed
Yest=zeros(L/2+1,1); %signal estimate of current block
Rnninv_initialized=0; %flag that is set to 1 when Rnninv is initialized
count=0; %counts number of blocks that have been processed
count2=0; %counts number of signal samples that have been processed

%%  create legend entries (for plotting)
if plotresults==1
for m=1:nbmicsnode(shownode)
    Leg{m}=['Filter applied to mic ' num2str(m)];
end
for m=nbmicsnode(shownode)+1:dimnode(shownode)
    Leg{m}=['Filter applied to broadcast channel #' num2str(m-nbmicsnode(shownode))];
end
screen_size = get(0, 'ScreenSize');
f1 = figure(1);
set(f1, 'Position', [0 0 screen_size(3) screen_size(4) ] );
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  WOLA-DANSE ALGORITHM
tic
for iter=1:L/2:lengthsignal-L
    count=count+1;
    count2=count2+L/2;
    
    %create z-signals
    for k=1:nbnodes
        Yblock{k}=fft(sqrt(Han(:,1:nbmicsnode(k))).*Y{k}(iter:iter+L-1,:)).';
        Yblock{k}=Yblock{k}(:,1:L/2+1);
        for u=1:L/2+1
            Zblock(k,u)=Wext{k}(:,u)'*Yblock{k}(:,u);
        end
    end
    
    %Determine if current block contains speech or not
    if onoff(iter)==1 %speech is active
        speech_active=1;
        Ryysamples=Ryysamples+1;
    else
        speech_active=0;
        Rnnsamples=Rnnsamples+1;
        if Rnninv_initialized==0 && Rnnsamples>max(nbmicsnode)+nbnodes-1 %initialize Rnninv when sufficient samples are available (only performed once)
            for k=1:nbnodes
                for u=1:L/2+1
                    Rnninv{k}(:,:,u)=inv(Rnn{k}(:,:,u));
                end
            end
            Rnninv_initialized=1;
        end
    end
    
    %check when the noise reduction filters can be updated (needs minimum
    %number of samples for both Ryy and Rnn)
    if startupdating==0 && (Ryysamples>max(nbmicsnode)+nbnodes-1 && Rnnsamples>max(nbmicsnode)+nbnodes-1) %only update internal filters if sufficient samples have been collected
        disp(['Filters started updating after ' num2str(iter/fs) ' seconds'])
        startupdating=1;
        Ryysamples=0;
        Rnnsamples=0;
    end
    
    %compute internal optimal filters at each node
    for k=1:nbnodes
        Zk=[Zblock(1:k-1,:); Zblock(k+1:end,:)]; %remove data of node k
        In{k}=[Yblock{k};Zk]; %inputs of node k
        for u=1:L/2+1
            
            if speech_active==1
                Ryy{k}(:,:,u)=lambda*Ryy{k}(:,:,u)+(1-lambda)*(In{k}(:,u)*In{k}(:,u)');
            else
                Rnn{k}(:,:,u)=lambda*Rnn{k}(:,:,u)+(1-lambda)*(In{k}(:,u)*In{k}(:,u)');
                if Rnninv_initialized==1
                    Rnninv{k}(:,:,u)=(1/lambda)*Rnninv{k}(:,:,u)-((Rnninv{k}(:,:,u)*In{k}(:,u))*(Rnninv{k}(:,:,u)*In{k}(:,u))')/((lambda^2/(1-lambda))+lambda*In{k}(:,u)'*Rnninv{k}(:,:,u)*In{k}(:,u));
                end
            end
            
            if startupdating==1 %do not update Wint in the beginning
            Rxx{k}(:,:,u)=Ryy{k}(:,:,u)-Rnn{k}(:,:,u);
            %%%%%%make Rxx rank 1%%%%%%%
            [X,D]=eig(Rxx{k}(:,:,u));
            D=real(D); %must be real (imaginary part due to numerical noise)
            [Dmax,maxind]=max(diag(D)); %find maximal eigenvalue
            Rxx{k}(:,:,u)=X(:,maxind)*abs(Dmax)*X(:,maxind)'; %Rxx is assumed to be rank 1 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            P=Rnninv{k}(:,:,u)*Rxx{k}(:,:,u);
            Wint{k}(:,u)= (1/(mu+trace(P)))*P(:,1); %Rank1-SDWMWF (see Cornelis et al., "Performance analysis of multichannel Wiener filter based noise reduction in hearing aids under second order statistics estimation errors", IEEE Trans. Audio, Speech and Language Processing, vol. 19, No. 5, July 2011)
            Wext{k}(:,u)= lambda_ext*Wext{k}(:,u)+(1-lambda_ext)*Wext_target{k}(:,u); %take small step towards Wext_target
            end
        end
        wscale{k}=diag(sqrt(diag(sum(Ryy{k},3)))); %how to scale the filters according to output power (for plotting, not used in the algorithm)    
    end
    
    % DANSE update of external filters
    if Ryysamples>=min_nb_samples && Rnnsamples>=min_nb_samples %only update when there are sufficient fresh samples for BOTH Ryy and Rnn
        if startupdating==0
            disp('min_nb_samples is smaller than max(nbmicsnode)+nbnodes-1, no updates will be performed')
        end
        
        %reset counters
        Ryysamples=0;
        Rnnsamples=0;
        nbupdates=nbupdates+1;
        
        %perform updates
        if simultaneous==0 %sequential node updating
            Wext_target{updatetoken}=(1-alpha)*Wext_target{updatetoken}+alpha*Wint{updatetoken}(1:nbmicsnode(updatetoken),:);            
            updatetoken=rem(updatetoken,nbnodes)+1; 
        elseif simultaneous==1 %simultaneous node updating
            for k=1:nbnodes
                Wext_target{k}=(1-alpha)*Wext_target{k}+alpha*Wint{k}(1:nbmicsnode(k),:);
             end
        end
        disp(['New target(s) for external filters after ' num2str(iter/fs) ' seconds'])
    end
    
    %compute node-specific output at all nodes
    for k=1:nbnodes
        for u=1:L/2+1
            Yest(u)=Wint{k}(:,u)'*In{k}(:,u);
        end
        blockest=real(ifft([Yest;conj(flipud(Yest(2:L/2)))]));
        estimation{k}(iter:iter+L-1)=estimation{k}(iter:iter+L-1)+sqrt(hann(L)).*blockest;
    end
    
    %compute time domain filters (not used, just for plotting)
    for k=1:nbnodes
        for q=1:dimnode(k)
            w{k}(:,q)=ifftshift(real(ifft(conj([Wint{k}(q,:) fliplr(conj(Wint{k}(q,2:L/2)))]))),2); %conj because we apply W^H to the signals, ifftshift to make them causal
        end
    end
    
    
    if 100*iter/length(Y{1}(:,1))>teller
        disp([num2str(teller) '% processed'])
        teller=teller+1;
        
        if plotresults==1 && startupdating==1
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %compute utility of all channels of 'shownode', and plot intermediate
        %results (CAN BE COMMENTED OUT IF NOT REQUIRED, or set 'plotresults' to 0)
        
        %More info on utility measures: See
        %
        %A. Bertrand and M. Moonen, "Efficient sensor subset selection and link
        %failure response for linear MMSE signal estimation in wireless sensor
        %networks", Proc. of the European signal processing conference
        %(EUSIPCO), Aalborg, Denmark, Aug. 2010, pp. 1092-1096.
        
            utility=zeros(dimnode(shownode),1);
            binrelativeutility=zeros(dimnode(shownode),1);
            totcost=0;
            W=Wint{shownode}';
            for u=2:size(Ryy{shownode},3)
                binutility=real((1./diag(pinv(Ryy{shownode}(:,:,u)))).*abs(W(u,:)').^2);
                utility=utility+binutility;
                bincost=real(Rxx{shownode}(1,1,u)-Rxx{shownode}(:,1,u)'*W(u,:).'); %value of MMSE cost for this particular bin
                if bincost<eps %ill conditioned computation
                    bincost=eps;
                    binutility=zeros(dimnode(shownode),1);
                end
                totcost=totcost+bincost;
                binrelativeutility=binrelativeutility+binutility/bincost;
                
            end
            subplot(3,1,1)
            bar(utility/totcost)
            title(['relative utility of each channel in node ' num2str(shownode)])
            subplot(3,1,2)
            plot(w{shownode}*wscale{shownode})
            legend(Leg{:})
            title(['applied filters in node ' num2str(shownode)])
            subplot(3,1,3)
            plot(1/fs*(1:size(estimation{shownode},1)),Y{shownode}(:,1));
            hold on
            plot(1/fs*(1:size(estimation{shownode},1)),estimation{shownode},'r')
            title([num2str(100*iter/length(Y{1})) '% processed'])
            hold off
            drawnow
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    % save
    if count2>saveperiod
        count2=0;
        disp([num2str(100*iter/length(Y{1})) '% processed'])
        save(fname,'estimation','lengthsignal','fs','nbupdates','L','onoff','Wext','Wint','w','updatetoken')
    end
end
save(fname,'estimation','lengthsignal','fs','nbupdates','L','onoff','Wext','Wint','w','updatetoken')
toc

 output=estimation;
end
