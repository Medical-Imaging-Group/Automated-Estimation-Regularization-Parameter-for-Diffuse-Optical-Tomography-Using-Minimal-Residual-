function [fwd_mesh,pj_error] = reconstruct_stnd_cw_GCV(fwd_mesh,...
						data_fn,...
						iteration,...
						output_fn,...
						filter_n)

% [fwd_mesh,pj_error] = reconstruct_stnd_cw(fwd_mesh,...
%                                        recon_basis,...
%                                        data_fn,...
%                                        iteration,...
%                                        lambda,...
%                                        output_fn,...
%                                        filter_n)
%                                            
% CW Reconstruction program for standard meshes 
% using MRM method for finding the regularization

% fwd_mesh is the input mesh (variable or filename)
% data_fn is the boundary data (variable or filename)
% iteration is the max number of iterations
% output_fn is the root output filename
% filter_n is the number of mean filters




% set modulation frequency to zero.
frequency = 0;

tic;
%****************************************
% If not a workspace variable, load mesh
if ischar(fwd_mesh)== 1
  fwd_mesh = load_mesh(fwd_mesh);
end
if ~strcmp(fwd_mesh.type,'stnd')
    errordlg('Mesh type is incorrect','NIRFAST Error');
    error('Mesh type is incorrect');
end

%*******************************************************
% read data - This is the calibrated experimental data or simulated data
anom = load_data(data_fn);
if ~isfield(anom,'paa')
    errordlg('Data not found or not properly formatted','NIRFAST Error');
    error('Data not found or not properly formatted');
end

% remove zeroed data
anom.paa(anom.link(:,3)==0,:) = [];
data_link = anom.link;

anom = anom.paa;
anom = log(anom(:,1)); %take log of amplitude
fwd_mesh.link = data_link;



%*******************************************************
% Initiate projection error
pj_error = [];
%*******************************************************
% Initiate log file
fid_log = fopen([output_fn '.log'],'w');
fprintf(fid_log,'Forward Mesh   = %s\n',fwd_mesh.name);
fprintf(fid_log,'Frequency      = %f MHz\n',frequency);
if ischar(data_fn) ~= 0
    fprintf(fid_log,'Data File      = %s\n',data_fn);
end
fprintf(fid_log,'Filter         = %d\n',filter_n);
fprintf(fid_log,'Output Files   = %s_mua.sol\n',output_fn);
fprintf(fid_log,'               = %s_mus.sol **CW recon only**\n',output_fn);
fprintf(fid_log,'Initial Guess mua = %d\n',fwd_mesh.mua(1));


for it = 1 : iteration
  
  % Calculate jacobian
  [J,data]=jacobian_stnd(fwd_mesh,frequency);
  data.amplitude(data_link(:,3)==0,:) = [];

  % Set jacobian as Phase and Amplitude part instead of complex
  J = J.complete;

  % Read reference data
  clear ref;
  ref = log(data.amplitude);
  
  data_diff = (anom-ref);

  pj_error = [pj_error sum(abs(data_diff.^2))]; 
  
  disp('---------------------------------');
  disp(['Iteration Number          = ' num2str(it)]);
  disp(['Projection error          = ' num2str(pj_error(end))]);

  fprintf(fid_log,'---------------------------------\n');
  fprintf(fid_log,'Iteration Number          = %d\n',it);
  fprintf(fid_log,'Projection error          = %f\n',pj_error(end));

  if it ~= 1
    p = (pj_error(end-1)-pj_error(end))*100/pj_error(end-1);
    disp(['Projection error change   = ' num2str(p) '%']);
    fprintf(fid_log,'Projection error change   = %f %%\n',p);
    if p <= 2
      disp('---------------------------------');
      disp('STOPPING CRITERIA REACHED');
      fprintf(fid_log,'---------------------------------\n');
      fprintf(fid_log,'STOPPING CRITERIA REACHED\n');
     break
    end
  end

  
  % Normalize Jacobian wrt optical values
  N = fwd_mesh.mua;
  nn = length(fwd_mesh.nodes);
  % Normalise by looping through each node, rather than creating a
  % diagonal matrix and then multiplying - more efficient for large meshes
  for i = 1 : nn
      J(:,i) = J(:,i).*N(i,1);
  end
  clear nn N  
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

  % Set bounds and intial guess for finding the regualrization
  if it == 1
      foo = fwd_mesh.mua;
      a = 0; b = 1000; % Regularization is Searched in [0,1000] interval
  end
  [U S V ] = svd(J);s = diag(S);
    bhat = U'*data_diff;
    lambda.value = fminbnd(@(lambda)...
                    TikGCVfun(lambda, bhat, s),...
                    a,b,optimset( 'TolX', 1e-16));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %   
  % build hessian
  [nrow,ncol]=size(J);


  Hess = zeros(ncol);
  Hess = (J'*J);

  reg_mua = lambda.value;
  reg = ones(ncol,1);
  reg = reg.*reg_mua;

  disp(['Mua Regularization        = ' num2str(reg(1,1))]);
  fprintf(fid_log,'Mua Regularization        = %f\n',reg(1,1));

  % Add regularisation to diagonal - looped rather than creating a matrix
  % as it is computational more efficient for large meshes
  for i = 1 : ncol
      Hess(i,i) = Hess(i,i) + reg(i);
  end

  % Calculate update
  foo = Hess\J'*data_diff; 
  
  foo = foo.*fwd_mesh.mua;
  
  % Update values
  fwd_mesh.mua = fwd_mesh.mua + foo;
  fwd_mesh.kappa = 1./(3.*(fwd_mesh.mua + fwd_mesh.mus));
  
  clear  Hess Hess_norm tmp data_diff G

  
  % We dont like -ve mua or mus! so if this happens, terminate
  if (any(fwd_mesh.mua<0) | any(fwd_mesh.mus<0))
    disp('---------------------------------');
    disp('-ve mua calculated...not saving solution');
    fprintf(fid_log,'---------------------------------\n');
    fprintf(fid_log,'STOPPING CRITERIA REACHED\n');
    break
  end
  
  % Filtering if needed!
  if filter_n > 1
    fwd_mesh = mean_filter(fwd_mesh,abs(filter_n));
  elseif filter_n < 0
    fwd_mesh = median_filter(fwd_mesh,abs(filter_n));
  end

  if it == 1
    fid = fopen([output_fn '_mua.sol'],'w');
  else
    fid = fopen([output_fn '_mua.sol'],'a');
  end
  fprintf(fid,'solution %g ',it);
  fprintf(fid,'-size=%g ',length(fwd_mesh.nodes));
  fprintf(fid,'-components=1 ');
  fprintf(fid,'-type=nodal\n');
  fprintf(fid,'%f ',fwd_mesh.mua);
  fprintf(fid,'\n');
  fclose(fid);
  
  if it == 1
    fid = fopen([output_fn '_mus.sol'],'w');
  else
    fid = fopen([output_fn '_mus.sol'],'a');
  end
  fprintf(fid,'solution %g ',it);
  fprintf(fid,'-size=%g ',length(fwd_mesh.nodes));
  fprintf(fid,'-components=1 ');
  fprintf(fid,'-type=nodal\n');
  fprintf(fid,'%f ',fwd_mesh.mus);
  fprintf(fid,'\n');
  fclose(fid);
end

% close log file!
time = toc;
fprintf(fid_log,'Computation TimeRegularization = %f\n',time);
fclose(fid_log);


function G = TikGCVfun(alpha, bhat, s)
%
%    G = TikGCVfun(alpha, bhat, s);
%
%  This function evaluates the GCV function for Tikhonov
%  regularization.  
%
%  Input:  alpha -  regularization parameter
%           bhat -  vector U'*b, where U = left singular vectors
%              s -  vector containing the singular values
%
%  Output:     G -  the scalar G(alpha).
%

n = length(bhat);

s2 = abs(s) .^ 2;
alpha2 = alpha^2;

t1 = 1 ./ (s2 + alpha2);
t2 = abs(bhat .* t1) .^2;

G = ( sum(t2) / n ) / ( sum(t1) / n )^2

