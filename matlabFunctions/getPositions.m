function positions = getPositions(varargin)
%GETPOSITIONS Summary of this function goes here
%   Detailed explanation goes here

    p = inputParser;
    addOptional(p,'km', eye(3));
    addOptional(p,'index', []);
    parse(p,varargin{:});

    positions = webread('http://147.32.86.182:5001/centers');
    if isnumeric(positions)
        positions = num2cell(positions', 1)';
    end
    found = false(size(positions, 1), 1);
    for ind=1:size(positions, 1)
        found(ind) = size(positions{ind},1) > 1;
    end
    
    if isempty(p.Results.index)
        positions = cell2mat(positions(found)');
        if isempty(positions)
            positions = zeros(3,0);
        end
    else
        positions(~found) = {[nan; nan; nan]};
        positions = cell2mat(positions(p.Results.index)');
    end
    positions = hom_transform(positions(1:2,:), p.Results.km)';
end

