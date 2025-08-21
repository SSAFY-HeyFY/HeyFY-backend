package com.ssafy.ssashinsa.heyfy.inquire.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.inquire.dto.ShinhanInquireSingleDepositResponseDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "예금주 계좌 상세 조회 (테스트용)", description = "단일 계좌 정보와 헤더를 포함한 전체 API 응답을 조회합니다. (백엔드 테스트용)")
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "성공적으로 계좌 정보를 조회했습니다.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ShinhanInquireSingleDepositResponseDto.class)
                )
        ),
        @ApiResponse(
                responseCode = "500",
                description = "서버 내부 오류",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class)
                )
        )
})
public @interface InquireSingleDepositTestDocs {
}
