function [ pats, patwh, im_array ] = image2patches( fname )
%IMAGE2PATCHES  Returns patches from the grayscaled image.
%
%   [ pat_struct, im_array ] = image2patches( fname )
%
%    -  pats is a cell array; each cell contains an 8x8 patch.
%
%    -  patwh contains the patch width and height of the image.
%
%    -  im_array is the array given by double(rgb2gray(imread(fname))),
%       truncated so the width and height is a multiple of 8.

    % Read image and truncate:
    im_array = double(rgb2gray(imread(fname)));
    % im_array = double(imread(fname));
    patwh = floor(size(im_array)/8);
    im_array = im_array(1:patwh(1)*8, 1:patwh(2)*8);
    
    % Get patches:
    pats = mat2cell(im_array, 8*ones(patwh(1),1), 8*ones(patwh(2),1));

end

