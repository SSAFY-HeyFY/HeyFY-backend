package com.ssafy.ssashinsa.heyfy.exchange.docs;

import com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate.RealTimeRateGroupResponseDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

import java.lang.annotation.*;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "현재 환율(USD | CNY | VND)")
@Documented
@ApiResponses({
        @ApiResponse(responseCode = "200", description = "조회 성공",
                content = @Content(schema = @Schema(implementation = RealTimeRateGroupResponseDto.class))),
})
public @interface ExchangeRateCurrentDocs {
    // 이 어노테이션은 환전 페이지 API의 Swagger 문서화에 사용됩니다.
    // 각 API 메소드에 적용하여 응답 코드와 예시를 정의합니다.
}
