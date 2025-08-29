package com.ssafy.ssashinsa.heyfy.exchange.docs;

import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.ExchangeResponseDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

import java.lang.annotation.*;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "한화->외화 환전")
@Documented
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "환전 성공 또는 핀 번호 오류",
                content = @Content(
                        schema = @Schema(implementation = ExchangeResponseDto.class)
                )
        )
})
public @interface ExchangeDocs {
}
