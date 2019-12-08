%% Initialization
clear; clc;
load('DCTQ.mat');
for j=1:5
    
    %% Read patches from image - see 'help image2patches':
    [pats_in, patwh, im_array] = image2patches(strcat(num2str(j), '.png'));
    imsz = size(im_array);  npx = prod(imsz);

    %% Process each patch:
    avgenergy = zeros(8);               % keep track of the average DCT energy
    nzeros = zeros(8);                  % count the number of nzeros

    npatsin = numel(pats_in);
    pats_rec = cell(npatsin,1);
    for i = 1:npatsin
            %% TO-DO: Process each patch in this loop!

            % encode
            X = pats_in{i};
            G = dct2(X - 128);
            avgenergy = avgenergy + G .^ 2 ./ npatsin;
            B = round(G ./ Q);
            nzeros(B~=0) = nzeros(B~=0) + 1;
            % decode
            pats_rec{i} = idct2(B .* Q) + 128;
    end

    %% Reconstruct image from patches and show results:
    im_rec = floor(cell2mat(reshape(pats_rec, patwh)));
    im_rec = im_rec - min(im_rec(:)); % shift data such that the smallest element of A is 0
    im_rec = im_rec / max(im_rec(:)); % normalize the shifted data to 1
    fname = fullfile('data', strcat(num2str(j), '.png'));
    imwrite(im_rec ,fname);
end