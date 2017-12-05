function [image, resolution] = RaspiImage(host, port)
%RASPIIMAGE Summary of this function goes here
%   Detailed explanation goes here
    sock = tcpclient(host, port);
    sock.write(uint8(['Matlab console client' 13 10]));
    image = [];
    resolution = typecast(sock.read(4), 'uint16');
    while(sock.BytesAvailable == 0)
        pause(0.001)
    end
    while(sock.BytesAvailable > 0)
        image = [image sock.read()];
        pause(0.05);
    end
    image = permute(reshape(image, [3,resolution]),[3,2,1]);
end

