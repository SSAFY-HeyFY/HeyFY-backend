package com.ssafy.ssashinsa.heyfy.exchange.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.ExchangeResponseDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.enums.ParameterIn;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

import java.lang.annotation.*;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "외화->한화 환전")
@Documented
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "환전 성공 또는 핀 번호 오류",
                content = @Content(
                        schema = @Schema(implementation = ExchangeResponseDto.class),
                        examples = {
                                @ExampleObject(
                                        name = "환전 성공",
                                        value = """
                                                {
                                                  "depositAccountBalance": "100.00",
                                                  "withdrawalAccountBalance": "500000.00",
                                                  "transactionBalance": "100.00",
                                                  "correct": true
                                                }
                                                """
                                ),
                                @ExampleObject(
                                        name = "핀 번호 오류",
                                        value = """
                                                {
                                                  "depositAccountBalance": null,
                                                  "withdrawalAccountBalance": null,
                                                  "transactionBalance": null,
                                                  "correct": false
                                                }
                                                """
                                )
                        }
                )
        ),
        @ApiResponse(responseCode = "400", description = "잘못된 요청",
                content = @Content(mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(name = "필수 정보 누락",
                                        ref = "#/components/examples/MissingRequired")
                        }))
})
@Parameter(name = "Authorization", description = "JWT 액세스 토큰 (Bearer <token>)", in = ParameterIn.HEADER, required = true)
@Parameter(name = "sid", description = "세션 ID", in = ParameterIn.HEADER, required = true)
public @interface ExchangeForeignDocs {
}
