function dx = fcn_general_dynamics(t, x, A, alpha, s, c0, rate, threshold, type)
%     switch type
%         case 'incremental_1'
%             A = diag(A).*eye(numel(x));
%             dx = A*(-x + f(x,s,alpha,threshold).*sigma_bar(s-1,alpha));%exp(-(alpha)*(s-1)));
%         case 'incremental_2'
%             A = diag(A).*eye(numel(x));
%             dx = f_A(x,A,alpha,c0,rate,threshold)*(-x + 1*f(x,s,alpha,threshold).*sigma_bar(s-1,alpha));%exp(-(alpha)*(s-1)));
%         case 'collective'
%             dx = A*(-x + 1);
%         case 'incremental_3'
%             A = diag(A).*eye(numel(x));
%             %dx = -trigger(x,t,s,alpha).*(A + Delta(x,s,A,alpha,rate))*x;
%             dx = -trigger(x,t,s,alpha).*diag(rate./(c0*sigma_bar(s-1,alpha)))*x;
%     end

    switch type
        case 'incremental_1'
            A  = diag(A).*eye(numel(x));
            dx = A*(-x + f(x,s,alpha,threshold).*sigma_bar(s-1,alpha));%exp(-(alpha)*(s-1)));
        case 'incremental_2'
            A  = diag(A).*eye(numel(x));
            dx = f_A(x,A,alpha,c0,rate,threshold)*(x + 1*f(x,s,alpha,threshold).*sigma_bar(s-1,alpha));%exp(-(alpha)*(s-1)));
        case 'incremental_3'
            A  = diag(A).*eye(numel(x));
            %dx = -trigger(x,t,s,alpha).*(A + Delta(x,s,A,alpha,rate))*x;
            dx = -trigger(x,t,s,alpha).*diag(rate./(c0*sigma_bar(s-1,alpha)))*x;
        case 'collective'
%             dx = (A*x).*(x>0);
            dx = (A*x);
            if any(x<0)
%                 disp('Reached limit!')
%                 dx(x<0) = 0;
%                   dx = (A*x).*(x>0);
            end
%             dx = A*(x + 0).*max(x,0);
        case 'collective_sat'
            dx = (A*x).*(x>0);
%             dx = (A*x).*tanh(1000*(x - threshold));
    end

end

% function out = f_A_col(x,A,alpha,c0,rate, threshold)
%     out = zeros(numel(x),numel(x));
%     if any(x<0)
%     out    
%     end
%     for j = 1:numel(x)
%         if j-1 == 0
%             out(j,j) = A(j,j);
%         else
%             delta    = x(j-1) - exp(-alpha*(j - 2));
%             if (abs(delta) < threshold)
%                 out(j,j) = rate(j)./(c0*sigma_bar(j-1,alpha));
%             else
%                 out(j,j) = A(j,j);
%             end
%         end
%     end
% %     transpose(diag(out))
% end


function out = f_A(x,A,alpha,c0,rate, threshold)
    out = zeros(numel(x),numel(x));
    for j = 1:numel(x)
        if j-1 == 0
            out(j,j) = A(j,j);
        else
            delta    = x(j-1) - exp(-alpha*(j - 2));
            if (abs(delta) < threshold)
                out(j,j) = rate(j)./(c0*sigma_bar(j-1,alpha));
            else
                out(j,j) = A(j,j);
            end
        end
    end
%     transpose(diag(out))
end

function out = f(x,s,alpha, threshold)
    out = zeros(numel(x),1);
    for j = 1:numel(x)
        if j-1 == 0
            out(j) = 1;
        else
            %delta  = abs(x(j-1) - exp(-(alpha)*((s(j)-1) - 1)));
            delta  = x(j-1) - exp(-alpha*(j - 2));
            out(j) = abs(delta) < threshold;
        end
    end
    if out(3) == 1
        test = 1;
    end
end

function out = trigger(x,t,s,alpha)
    c0 = 100;
    out = zeros(numel(x),1);
    for j = 1:numel(x)
        if j-1 == 0
            out(j) = 1;
        else
%             out(j) = (x(j-1)<0.001);
%             t_cum = cumsum(c0*sigma_bar(s(j)-1,alpha));
            t_cum = cumsum(c0*sigma_bar(s-1,alpha));
%             out(j) = t>(c0*sigma_bar(s(j)-1,alpha));
            out(j) = t>t_cum(j-1);
        end
    end
end

function out = Delta(x,s,A,alpha,rate)
    c0  = 100;
    out = zeros(numel(x), numel(x));
    for j = 1:numel(x)
        if j-1 == 0
            out(j,j) = 0;
        else
            out(j,j)  = rate(j)./(c0*sigma_bar(s(j)-1,alpha)) - A(j,j);
        end
    end
end

function out = sigma_bar(Nj,alpha)
    out   = exp(-alpha*Nj);
end
