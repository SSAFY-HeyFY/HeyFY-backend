package com.ssafy.ssashinsa.heyfy.exchange.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.AccountBalanceResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate.ExchangeRateGroupDto;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

import java.lang.annotation.*;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@ApiResponses({
        @ApiResponse(responseCode = "200", description = "조회 성공",
                content = @Content(schema = @Schema(implementation = AccountBalanceResponseDto.class))),
        @ApiResponse(responseCode = "400", description = "잘못된 요청",
                content = @Content(mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(name = "계좌 정보 없음",
                                        value = "{\n" +
                                                "  \"status\": 400,\n" +
                                                "  \"httpError\": \"BAD_REQUEST\",\n" +
                                                "  \"errorCode\": \"ACCOUNT_NOT_FOUND\",\n" +
                                                "  \"message\": \"Account not found\"\n" +
                                                "}"
                                )
                        }))
})
public @interface AccountBalanceDocs {
}
