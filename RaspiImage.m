function [image, resolution] = RaspiImage(host, port, object, channel)
%RASPIIMAGE Summary of this function goes here
%   Detailed explanation goes here
    sock = tcpclient(host, port);
    sock.write(uint8(sprintf('GET %s %s\r\n', object, channel)));
    resolution = typecast(sock.read(6), 'uint16');
    
    image = [];
    
    if all(resolution)
        while(sock.BytesAvailable == 0)
            pause(0.001)
        end
        while(sock.BytesAvailable > 0)
            image = [image sock.read()];
            pause(0.05);
        end
    end
    
    image = permute(reshape(image, [resolution([3 2 1])]),[3,2,1]);
end

