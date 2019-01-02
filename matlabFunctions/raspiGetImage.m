function varargout = raspiGetImage(host, object, channel)
%RASPIIMAGE Summary of this function goes here
%   Detailed explanation goes here
    if nargin < 1
        error("Host must be first argument")
    elseif nargin < 2
        path = sprintf("http://%s:5001/image", host);
    elseif nargin < 3
        path = sprintf("http://%s:5001/image/%s/image", host, object);
    elseif nargin < 4
        path = sprintf("http://%s:5001/image/%s/%s", host, object, channel);
    end
    
    image = webread(path);
    if nargout > 0
        varargout{1} = image;
    else
        imshow(image);
    end
end

