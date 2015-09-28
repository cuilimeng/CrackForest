function ScaledMatrix = ScaledMatrixByColumn( OriginalMatrix, MinValue, MaxValue)

matrix = OriginalMatrix';

para.ymin = MinValue;
para.ymax = MaxValue;

resultTransposition = mapminmax( matrix, para );

ScaledMatrix = resultTransposition';
    
end
 
    
