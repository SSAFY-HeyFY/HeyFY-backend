package com.ssafy.ssashinsa.heyfy.exchange.docs;

import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.HistoricalAnalysisResponseDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

import java.lang.annotation.*;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "환율 기록 분석 조회")
@Documented
@ApiResponses({
        @ApiResponse(responseCode = "200", description = "조회 성공",
                content = @Content(schema = @Schema(implementation = HistoricalAnalysisResponseDto.class)))
})
public @interface HistoricalAnalysisDocs {
}
