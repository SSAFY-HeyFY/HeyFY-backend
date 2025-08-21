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
        @ApiResponse(responseCode = "200", description = "환전 성공",
                content = @Content(schema = @Schema(implementation = ExchangeResponseDto.class))),
        @ApiResponse(responseCode = "400", description = "잘못된 요청",
                content = @Content(mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(name = "필수 정보 누락",
                                        ref = "#/components/examples/MissingRequired")
                        }))
})
public @interface ExchangeForeignDocs {
}
