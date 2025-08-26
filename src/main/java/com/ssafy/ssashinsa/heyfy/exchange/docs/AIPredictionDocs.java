package com.ssafy.ssashinsa.heyfy.exchange.docs;

import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.AIPredictionResponseDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

import java.lang.annotation.*;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "AI 예측 조회 (미사용)")
@Documented
@ApiResponses({
        @ApiResponse(responseCode = "200", description = "조회 성공",
                content = @Content(schema = @Schema(implementation = AIPredictionResponseDto.class)))
})
public @interface AIPredictionDocs {
}
