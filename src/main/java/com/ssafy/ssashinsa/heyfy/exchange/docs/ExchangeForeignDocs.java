package com.ssafy.ssashinsa.heyfy.exchange.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.ExchangeResponseDto;
import io.swagger.v3.oas.annotations.Operation;
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
        @ApiResponse(
                responseCode = "400",
                description = "잘못된 요청",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "필수 정보 누락",
                                        ref = "#/components/examples/MissingRequired"
                                )
                        }
                )
        ),
        @ApiResponse(
                responseCode = "403",
                description = "거래 잠금",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "거래 잠금",
                                        value = """
                                                {
                                                  "status": 403,
                                                  "httpError": "FORBIDDEN",
                                                  "errorCode": "TRADE_LOCKED",
                                                  "message": "Trading is temporarily locked. Please try again after a while."
                                                }
                                                """
                                )
                        }
                )
        ),
        @ApiResponse(
                responseCode = "429",
                description = "핀 시도 초과",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "핀 시도 초과",
                                        value = """
                                                {
                                                  "status": 429,
                                                  "httpError": "TOO_MANY_REQUESTS",
                                                  "errorCode": "PIN_TRADE_ATTEMPTS_EXCEEDED",
                                                  "message": "PIN authentication failed 5 times. Please try again after 30 seconds."
                                                }
                                                """
                                )
                        }
                )
        )
})
public @interface ExchangeForeignDocs { }
