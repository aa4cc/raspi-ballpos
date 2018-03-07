classdef RaspiBallPos < matlab.System ...
        & coder.ExternalDependency ...
        & matlab.system.mixin.Propagates ...
        & matlab.system.mixin.CustomIcon
    %
    % System object for readin position in image coordinates from
    % raspi-ballpos script.
    % 
    
    % Copyright 2016 The MathWorks, Inc.
    %#codegen
    %#ok<*EMCA>
    
    properties
        % Public, tunable properties.
    end
    
    properties (Nontunable)
        % Key to shared memory segment
        shm_key = 3145914;  
        % Framerate
        framerate = 50;
        % Number of frames
        N = 1000;
        % path Path of camera script
        path = '';
        % Number of objects
        objects = 1;
    end
    
    properties (Nontunable, Logical)
        % Execute raspi-ballpos script (not functional yet)
        execScript = false;
        % Concatenate the outputs
        concatenateOutputs = false;
        % Enable rotation
        rotation = false;
    end
    
    properties (Access = private)
        % Pre-computed constants.
    end
    
    methods
        % Constructor
        function obj = RaspiBallPos(varargin)
            coder.allowpcode('plain');
            % Support name-value pair arguments when constructing the object.
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access=protected)
        function setupImpl(obj) %#ok<MANU>
            if isempty(coder.target)
                % Place simulation setup code here
            else
                % Call C-function implementing device initialization
                coder.cinclude('raspiballpos.h')
                coder.ceval('raspiballpos_init', obj.shm_key, obj.execScript, [obj.path 0], obj.framerate, obj.N, obj.objects);
            end
        end
        
        function varargout = stepImpl(obj) 
            positions = zeros(3, obj.objects, 'single');
            if isempty(coder.target)
                % Place simulation output code here 
            else
                % Call C-function implementing device output
                coder.ceval('read_pos', ...
                    coder.wref(positions), ...
                    obj.objects);
            end
            
            if obj.rotation
                part = double(positions);
            else
                part = double(positions(1:2,:));
            end
            
            if obj.concatenateOutputs
                varargout{1} = part(:);
            else
                for k=1:obj.objects
                    varargout{k} = part(:,k);
                end
            end
        end
        
        function releaseImpl(obj) %#ok<MANU>
            if isempty(coder.target)
                % Place simulation termination code here
            else
                % Call C-function implementing device termination
                coder.ceval('raspiballpos_terminate');
            end
        end
    end
    
    methods (Access=protected)
        %% Define input properties
        function num = getNumInputsImpl(~)
            num = 0;
        end
        
        function num = getNumOutputsImpl(obj)
            if obj.concatenateOutputs
                num = 1;
            else
                num = obj.objects;
            end
        end
        
                
        function flag = isOutputSizeLockedImpl(~,~)
            flag = true;
        end
        
        function varargout = isOutputFixedSizeImpl(obj,~)
            if obj.concatenateOutputs
                varargout{1} = true;
            else
                for k = 1:obj.objects
                   varargout{k} = true;
                end
            end            
        end
        
        function icon = getIconImpl(~)
            % Define a string as the icon for the System block in Simulink.
            icon = 'RaspiBallPos';
        end
        
        function varargout = isOutputComplexImpl(obj)
            if obj.concatenateOutputs
                varargout{1} = false;
            else
                for k = 1:obj.objects
                   varargout{k} = false;
                end
            end            
        end
        
        function varargout = getOutputSizeImpl(obj)
            if obj.rotation
                n=3;
            else
                n=2;
            end
            if obj.concatenateOutputs
                varargout{1} = n*obj.objects;
            else
                for k = 1:obj.objects
                   varargout{k} = n;
                end
            end
        end
        
        function varargout = getOutputDataTypeImpl(obj)
            if obj.concatenateOutputs
                varargout{1} = 'double';
            else
                for k = 1:obj.objects
                   varargout{k} = 'double';
                end
            end
        end
        
        function varargout = getOutputNamesImpl(obj)
            if obj.concatenateOutputs && obj.rotation
                varargout{1} = 'Position [x1;y1;r1;x2...]';
            elseif obj.concatenateOutputs
                varargout{1} = 'Position [x1;y1;x2...]';
            else
                if obj.rotation
                    str = 'Position %d [x;y;r]';
                else
                    str = 'Position %d [x;y]';
                end
                
                for k = 1:obj.objects
                   varargout{k} = sprintf(str, k);
                end
            end
        end        
    end
    
    methods (Static, Access=protected)
        function simMode = getSimulateUsingImpl(~)
            simMode = 'Interpreted execution';
        end
        
        function isVisible = showSimulateUsingImpl
            isVisible = false;
        end
    end
    
    methods (Static)
        function name = getDescriptiveName()
            name = 'RaspiBallPos';
        end
        
        function b = isSupportedContext(context)
            b = context.isCodeGenTarget('rtw');
        end
        
        function updateBuildInfo(buildInfo, context)
            if context.isCodeGenTarget('rtw')
                % Update buildInfo
                srcDir = fullfile(fileparts(mfilename('fullpath')),'src');
                includeDir = fullfile(fileparts(mfilename('fullpath')),'include');
                addIncludePaths(buildInfo,includeDir);
                % Use the following API's to add include files, sources and
                % linker flags
                addIncludeFiles(buildInfo,'raspiballpos.h',includeDir);
                addSourceFiles(buildInfo,'raspiballpos.c',srcDir);
%                 addLinkFlags(buildInfo,{'-lwiringPi'});
                %addLinkObjects(buildInfo,'sourcelib.a',srcDir);
                %addCompileFlags(buildInfo,{'-D_DEBUG=1'});
                %addDefines(buildInfo,'MY_DEFINE_1')
            end
        end
    end
end
